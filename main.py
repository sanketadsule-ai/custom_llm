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
from cachetools import LRUCache

# -----------------------------
# 1. High-Performance Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-ultra-low-latency")

# API Version 2024-08-01-preview is the most stable for GPT-4o in 2026
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# LRU Cache for 0ms repeat response latency
RESPONSE_CACHE = LRUCache(maxsize=200)

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
# 3. Optimized Logic
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    # Security check
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # FAST CACHE: Check for exact repeat user context
        user_context = "".join([m.content for m in req.messages if m.role == "user"][-2:])
        ckey = hashlib.md5(user_context.encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        try:
            # GPT-4O OPTIMIZATION: 
            # 1. Reduced message window to 10 for better speed.
            # 2. Removed 'extra_body' to fix the 400 Bad Request error.
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

            # 10s timeout: GPT-4o is fast; if it takes longer, something is wrong.
            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=10.0 
            )

            first_chunk = True
            async for chunk in response:
                # Direct serialization to bytes for lower overhead
                chunk_data = chunk.model_dump(exclude_none=True)
                yield b"data: " + orjson.dumps(chunk_data) + b"\n\n"

                # Extract content for local cache
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        collected.append(delta.content)

                # TURBO-FLUSH: Force transmission on the very first token
                if first_chunk:
                    first_chunk = False
                    await asyncio.sleep(0) # Micro-pause to flush TCP buffer

            # Update cache in the background
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
            "X-Accel-Buffering": "no", # Critical for Real-time
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    import uvicorn
    import sys

    # Adaptive loop selection (prevents Windows errors, uses uvloop on Railway)
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