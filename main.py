import os
import hashlib
import logging
import asyncio
import orjson
import httpx
from typing import List, Optional, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ConfigDict
from openai import AsyncAzureOpenAI
from fastapi.responses import StreamingResponse
from cachetools import LRUCache

# -----------------------------
# 1. Setup & Connection Pooling
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-ultra")

# Persistent client for connection pooling
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    timeout=60.0
)

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
    http_client=http_client
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")
RESPONSE_CACHE = LRUCache(maxsize=500)

app = FastAPI()

# -----------------------------
# 2. Optimized Models
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
async def execute_tool_background(tool_data_list: List[dict]):
    """ Processes the tools in the background. """
    for tool in tool_data_list:
        name = tool.get("name")
        args = tool.get("args")
        # This will now only log ONCE per turn
        logger.info(f"⚡ [SINGLE EXEC] {name} | Args: {args}")

# -----------------------------
# 4. Main Bridge
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # CACHE KEY
        user_id = request.headers.get("x-user-id", "anonymous")
        user_context = "".join([m.content for m in req.messages if m.role == "user"][-5:])
        ckey = hashlib.md5(f"{user_id}:{user_context}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        # --- THE FIX: TURN-BASED STATE ---
        collected_text = []
        tool_buffer = {}
        has_executed_tools = False # STRICT ONE-SHOT LOCK

        try:
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in req.messages[-10:]],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": req.max_tokens,
                "stream_options": {"include_usage": True},
            }
            if req.tools:
                kwargs["tools"] = req.tools
                kwargs["tool_choice"] = "auto"

            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=10.0 
            )

            async for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                
                # 1. Accumulate Tool Chunks
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_buffer:
                            tool_buffer[idx] = {"name": "", "args": ""}
                        if tc.function.name:
                            tool_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_buffer[idx]["args"] += tc.function.arguments

                # 2. TRIGGER ON FINISH (WITH LOCK)
                # If finish_reason is 'tool_calls' and we haven't fired yet, DO IT.
                if finish_reason == "tool_calls" and not has_executed_tools:
                    if tool_buffer:
                        asyncio.create_task(execute_tool_background(list(tool_buffer.values())))
                        has_executed_tools = True # LOCK ENGAGED

                # 3. Stream Text
                if delta.content:
                    collected_text.append(delta.content)
                    yield b"data: " + orjson.dumps(chunk.model_dump(exclude_none=True)) + b"\n\n"

            if collected_text:
                RESPONSE_CACHE[ckey] = "".join(collected_text)

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ Maya Stream Error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )

if __name__ == "__main__":
    import uvicorn
    import sys
    loop_type = "uvloop" if sys.platform != "win32" else "asyncio"
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), 
                loop=loop_type, http="httptools", workers=1, access_log=False)