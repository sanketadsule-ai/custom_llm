import os
import hashlib
import logging
import asyncio
import orjson
from typing import List, Optional, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ConfigDict
from openai import AsyncAzureOpenAI
from fastapi.responses import StreamingResponse
from cachetools import TTLCache 

# -----------------------------
# 1. Setup & Environment
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-turbo-bridge")

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# Session-isolated cache (20-minute TTL)
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=1200)

app = FastAPI()

# -----------------------------
# 2. Data Models
# -----------------------------
class Message(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    tools: Optional[List[Any]] = None
    stream: bool = True
    max_tokens: int = 150

# -----------------------------
# 3. Background Tool Logic
# -----------------------------
async def execute_tool_silently(full_tool_calls: List[dict]):
    """
    Executes the completed tool JSON in the background.
    """
    for tc in full_tool_calls:
        name = tc.get("name")
        args = tc.get("args")
        # Replace this log with your actual n8n/webhook call
        logger.info(f"⚡ [SINGLE EXEC] Tool: {name} | Args: {args}")

# -----------------------------
# 4. Optimized Event Generator
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # CACHE KEY: ID + Text + Index
        msg_history = req.messages
        msg_count = len(msg_history)
        user_id = request.headers.get("x-user-id", "default_donor") 
        last_user_msg = "".join([m.content for m in msg_history if m.role == "user"][-5:])
        ckey = hashlib.md5(f"{user_id}:{last_user_msg}:{msg_count}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            logger.info("🚀 Cache Hit")
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected_content = []
        tool_accumulator = {}
        tool_executed = False  # THE GUARD: Prevents duplicate firing

        try:
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in msg_history[-10:]],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 100,
                "stream_options": {"include_usage": True},
            }
            if req.tools:
                kwargs["tools"] = req.tools

            response = await client.chat.completions.create(**kwargs)

            async for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # A. Accumulate Fragments
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_accumulator:
                            tool_accumulator[idx] = {"name": "", "args": ""}
                        if tc_chunk.function.name:
                            tool_accumulator[idx]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tool_accumulator[idx]["args"] += tc_chunk.function.arguments

                # B. One-Shot Trigger (The Fix for Redundant Calls)
                if finish_reason == "tool_calls" and not tool_executed:
                    if tool_accumulator:
                        asyncio.create_task(execute_tool_silently(list(tool_accumulator.values())))
                        tool_executed = True # Lock execution for this turn

                # C. Stream Content
                if delta.content:
                    collected_content.append(delta.content)
                    yield b"data: " + orjson.dumps(chunk.model_dump()) + b"\n\n"

            if collected_content:
                RESPONSE_CACHE[ckey] = "".join(collected_content)

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    import sys
    loop_type = "uvloop" if sys.platform != "win32" else "asyncio"
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), 
                loop=loop_type, http="httptools", workers=1, access_log=False)