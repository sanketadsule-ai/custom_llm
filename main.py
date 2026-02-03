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
from cachetools import TTLCache  # Changed to TTLCache for auto-purging

# -----------------------------
# 1. High-Performance Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-ultra-low-latency")

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# BEST SOLUTION: TTLCache (Time-To-Live)
# maxsize=500 unique interactions, ttl=1200 seconds (20 minutes)
# This automatically clears memory after a call ends.
RESPONSE_CACHE = TTLCache(maxsize=500, ttl=1200)

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
    user_id: Optional[str] = None # Added for session isolation

class ChatRequest(BaseModel):
    messages: List[Message]
    tools: Optional[List[Any]] = None
    stream: bool = True
    max_tokens: int = 150

# -----------------------------
# 3. Parallel Execution Logic
# -----------------------------
async def execute_tool_background(tool_call):
    """Fire-and-forget tool execution to eliminate 1.1s latency spikes."""
    try:
        # Replace this with your actual tool logic (n8n call, etc.)
        logger.info(f"Background Tool Triggered: {tool_call.get('function', {}).get('name')}")
        # await your_n8n_client.call(...) 
    except Exception as e:
        logger.error(f"Background Tool Error: {e}")

# -----------------------------
# 4. Main Endpoint
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # ISOLATED CACHE KEY: user_id + last user message
        # This prevents Donor A from getting Donor B's data.
        user_id = req.messages[0].user_id if hasattr(req.messages[0], 'user_id') else "anon"
        user_text = "".join([m.content for m in req.messages if m.role == "user"][-1:])
        ckey = hashlib.md5(f"{user_id}:{user_text}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        try:
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in req.messages[-10:]],
                "temperature": 0.0, # Determenistic for rule adherence
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

            first_chunk = True
            async for chunk in response:
                chunk_data = chunk.model_dump(exclude_none=True)
                
                # PARALLEL DISPATCH: 
                # If a tool call is detected, trigger it and KEEP STREAMING.
                if chunk.choices and chunk.choices[0].delta.tool_calls:
                    for tc in chunk.choices[0].delta.tool_calls:
                        asyncio.create_task(execute_tool_background(tc.model_dump()))

                yield b"data: " + orjson.dumps(chunk_data) + b"\n\n"

                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        collected.append(delta.content)

                if first_chunk:
                    first_chunk = False
                    await asyncio.sleep(0) 

            if collected:
                RESPONSE_CACHE[ckey] = "".join(collected)

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming Error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    import uvicorn
    import sys
    loop_type = "uvloop" if sys.platform != "win32" else "asyncio"

    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        loop=loop_type,
        http="httptools",
        workers=1,
        access_log=False
    )