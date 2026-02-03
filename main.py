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
# 1. High-Performance Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-ultra")

# PERFORMANCE: Use a shared HTTPX client for connection pooling.
# This prevents opening a new TCP connection for every request (~150ms savings).
limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
http_client = httpx.AsyncClient(limits=limits, timeout=60.0)

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
    http_client=http_client # Injecting optimized pool
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# LRU Cache: Store up to 500 unique user-context responses.
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
# 3. Tool Execution Logic
# -----------------------------
async def execute_tool_background(name: str, args: str):
    """ Executes tools like language_detection without blocking Maya's voice. """
    try:
        logger.info(f"⚡ [EXEC] {name} | Args: {args}")
        # Your internal webhook or tool logic goes here
    except Exception as e:
        logger.error(f"Tool Error: {e}")

# -----------------------------
# 4. Main Bridge Logic
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    # Security check
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # PER-USER CACHE FIX: 
        # We hash the user_id + the last few messages to ensure privacy.
        user_id = request.headers.get("x-user-id", "anonymous")
        user_context = "".join([m.content for m in req.messages if m.role == "user"][-5:])
        ckey = hashlib.md5(f"{user_id}:{user_context}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected_text = []
        executed_tool_ids = set() # PREVENTS DUPLICATE TOOL CALLS
        tool_buffer = {}          # Reconstructs tool arguments from stream chunks

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

            # 8s timeout: GPT-4o is fast; if it hangs, it's a network issue.
            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=8.0 
            )

            async for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                
                # A. Handle Tool Call Streaming (Prevents Duplicates)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_buffer:
                            tool_buffer[idx] = {"id": tc.id, "name": "", "args": ""}
                        
                        if tc.function.name: tool_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments: tool_buffer[idx]["args"] += tc.function.arguments

                # B. Trigger Tool on 'finish_reason'
                if chunk.choices[0].finish_reason == "tool_calls":
                    for idx, data in tool_buffer.items():
                        unique_id = data["id"] or f"idx_{idx}"
                        if unique_id not in executed_tool_ids:
                            asyncio.create_task(execute_tool_background(data["name"], data["args"]))
                            executed_tool_ids.add(unique_id)

                # C. Handle Regular Text Content
                if delta.content:
                    collected_text.append(delta.content)
                    # Use orjson for faster serialization than standard json
                    yield b"data: " + orjson.dumps(chunk.model_dump(exclude_none=True)) + b"\n\n"

            # Update cache after successful stream
            if collected_text:
                RESPONSE_CACHE[ckey] = "".join(collected_text)

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream Error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no", # Prevents proxy buffering (for real-time)
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    import sys

    # Uses uvloop on Linux (Railway/Docker) for massive performance gains
    loop_type = "uvloop" if sys.platform != "win32" else "asyncio"

    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        loop=loop_type,
        http="httptools", # High-performance HTTP parser
        workers=1,
        access_log=False
    )