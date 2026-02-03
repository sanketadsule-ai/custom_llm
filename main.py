import os
import hashlib
import logging
import asyncio
import orjson # 10x faster than standard json
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

# Uses the latest 2026-ready API version for Prompt Caching support
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# Local LRU cache for absolute repeat hits (instant 0ms latency)
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
    # Security check (Fast header comparison)
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # Check local cache first
        user_context = "".join([m.content for m in req.messages if m.role == "user"][-2:])
        ckey = hashlib.md5(user_context.encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        try:
            # OPTIMIZATION: Prepare kwargs with Prompt Caching hints
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in req.messages[-7:]], # Keep context short
                "temperature": 0.0,
                "stream": True,
                "max_tokens": req.max_tokens,
                "stream_options": {"include_usage": True},
                "extra_body": {
                    # Explicit hint for Azure GPT-5 Extended Prompt Caching
                    "prompt_cache_retention": "in_memory" 
                }
            }
            if req.tools:
                kwargs["tools"] = req.tools
                kwargs["tool_choice"] = "auto"

            # Execute with a tight timeout for voice responsiveness
            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=15.0 
            )

            first_chunk = True
            async for chunk in response:
                # OPTIMIZATION: Skip redundant dict conversions, use model_dump for speed
                chunk_data = chunk.model_dump(exclude_none=True)
                
                if not first_chunk: # Yield second chunk and onwards
                    if chunk.choices and chunk.choices[0].delta.content:
                        collected.append(chunk.choices[0].delta.content)
                
                yield b"data: " + orjson.dumps(chunk_data) + b"\n\n"

                # THE TURBO-FLUSH: Force packet transmission on first token
                if first_chunk:
                    if chunk.choices and chunk.choices[0].delta.content:
                        collected.append(chunk.choices[0].delta.content)
                    first_chunk = False
                    await asyncio.sleep(0) # Micro-yield to flush socket

            # Background: Update LRU cache
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
            "X-Accel-Buffering": "no", # Prevents proxy buffering
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    import uvicorn
    # ENGINE OPTIMIZATION: Use uvloop and httptools for 2026 concurrency
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        loop="uvloop",
        http="httptools",
        workers=1,
        access_log=False # Significant speedup by disabling logs per chunk
    )