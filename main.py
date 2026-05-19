# import os
# import hashlib
# import logging
# import asyncio
# import orjson
# import httpx
# from typing import List, Optional, Any
# from fastapi import FastAPI, Request, HTTPException
# from pydantic import BaseModel, ConfigDict
# from openai import AsyncAzureOpenAI
# from fastapi.responses import StreamingResponse
# from cachetools import LRUCache

# # -----------------------------
# # 1. High-Performance Setup
# # -----------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("maya-ultra-low-latency")

# limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
# http_client = httpx.AsyncClient(limits=limits, timeout=60.0)

# client = AsyncAzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version="2024-08-01-preview",
#     http_client=http_client
# )

# DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# RESPONSE_CACHE = LRUCache(maxsize=200)

# app = FastAPI()

# # -----------------------------
# # 2. Optimized Models
# # -----------------------------
# class Message(BaseModel):
#     model_config = ConfigDict(populate_by_name=True)
#     role: str
#     content: Optional[str] = None
#     tool_calls: Optional[List[Any]] = None
#     tool_call_id: Optional[str] = None
#     name: Optional[str] = None

# class ChatRequest(BaseModel):
#     messages: List[Message]
#     tools: Optional[List[Any]] = None
#     stream: bool = True
#     max_tokens: int = 100

# # -----------------------------
# # 3. Optimized Logic
# # -----------------------------
# @app.post("/custom-llm/chat/completions")
# async def chat_completions(req: ChatRequest, request: Request):
#     if request.headers.get("x-api-key") != AUTH_KEY:
#         raise HTTPException(status_code=401)

#     async def event_generator():
#         user_id = request.headers.get("x-user-id", "anonymous")
#         user_context = "".join([m.content for m in req.messages if m.role == "user"][-10:])
#         ckey = hashlib.md5(f"{user_id}:{user_context}".encode()).hexdigest()
        
#         if ckey in RESPONSE_CACHE:
#             yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
#             yield b"data: [DONE]\n\n"
#             return

#         collected = []
#         try:
#             # 1. Separate System Message
#             system_msg = next((m for m in req.messages if m.role == "system"), None)
            
#             # 2. Extract History (Excluding System)
#             history_pool = [m for m in req.messages if m.role != "system"]
            
#             # 3. Slicing with Tool Integrity
#             # We take the last 10 messages, but check if the first one is a 'tool'
#             slice_index = -10
#             if abs(slice_index) < len(history_pool):
#                 # If the first message in our slice is a 'tool', we MUST include the one before it
#                 if history_pool[slice_index].role == "tool":
#                     slice_index -= 1 
            
#             recent_history = history_pool[slice_index:]
            
#             # 4. Reconstruct final payload
#             final_messages = []
#             if system_msg:
#                 final_messages.append(system_msg.model_dump(exclude_none=True))
            
#             final_messages.extend([m.model_dump(exclude_none=True) for m in recent_history])

#             kwargs = {
#                 "model": DEPLOYMENT,
#                 "messages": final_messages,
#                 "temperature": 0.0,
#                 "stream": True,
#                 "max_tokens": req.max_tokens,
#                 "stream_options": {"include_usage": True},
#             }
            
#             if req.tools:
#                 kwargs["tools"] = req.tools
#                 kwargs["tool_choice"] = "auto"

#             response = await asyncio.wait_for(
#                 client.chat.completions.create(**kwargs),
#                 timeout=15.0 # Increased slightly for tool-heavy processing
#             )

#             first_chunk = True
#             async for chunk in response:
#                 chunk_data = chunk.model_dump(exclude_none=True)
#                 yield b"data: " + orjson.dumps(chunk_data) + b"\n\n"

#                 if chunk.choices and len(chunk.choices) > 0:
#                     delta = chunk.choices[0].delta
#                     if delta.content:
#                         collected.append(delta.content)

#                 if first_chunk:
#                     first_chunk = False
#                     await asyncio.sleep(0) 

#             if collected:
#                 RESPONSE_CACHE[ckey] = "".join(collected)

#             yield b"data: [DONE]\n\n"

#         except Exception as e:
#             logger.error(f"Streaming Error: {e}")
#             # Ensure the stream closes cleanly on error
#             yield b"data: [DONE]\n\n"

#     return StreamingResponse(
#         event_generator(), 
#         media_type="text/event-stream",
#         headers={
#             "X-Accel-Buffering": "no",
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive"
#         }
#     )

# @app.on_event("shutdown")
# async def shutdown_event():
#     await http_client.aclose()

# if __name__ == "__main__":
#     import uvicorn
#     import sys
#     loop_type = "uvloop" if sys.platform != "win32" else "asyncio"
#     uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), loop=loop_type)




import os
import hashlib
import logging
import asyncio
import orjson
import httpx
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ConfigDict
from openai import AsyncAzureOpenAI
from fastapi.responses import StreamingResponse
from cachetools import TTLCache

# -----------------------------
# 1. High-Performance Setup
# -----------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("maya-ultra-low-latency")

# Optimized connection pool for Azure
limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
http_client = httpx.AsyncClient(limits=limits, timeout=30.0)

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    http_client=http_client
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# TTLCache: auto-expires stale responses after 5 minutes
RESPONSE_CACHE = TTLCache(maxsize=200, ttl=300)

# Sentence boundary characters for voice-optimized flushing
FLUSH_CHARS = {".", "!", "?", ","}

app = FastAPI()


# -----------------------------
# 2. Models
# -----------------------------
class Message(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    role: str
    content: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = True
    max_completion_tokens: int = 80  # Tuned for voice: most spoken answers are 30-80 tokens


# -----------------------------
# 3. Connection Pre-Warming
# -----------------------------
@app.on_event("startup")
async def warmup():
    """Pre-establish connection to Azure so the first real request isn't cold."""
    try:
        await client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=1
        )
        logger.warning("Azure connection pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")


# -----------------------------
# 4. Optimized Streaming Endpoint
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    # Fast-path auth check
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # Cache key: hash of last full user+assistant exchange for better context sensitivity
        relevant = [m for m in req.messages if m.content][-4:]
        user_context = "".join(f"{m.role}:{m.content}" for m in relevant)
        ckey = hashlib.md5(user_context.encode()).hexdigest()

        # Cache hit: stream cached response instantly
        if ckey in RESPONSE_CACHE:
            cached_val = RESPONSE_CACHE[ckey]
            yield b"data: " + orjson.dumps({
                "choices": [{"delta": {"content": cached_val}}]
            }) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        sentence_buffer = ""

        try:
            # Build message list directly — avoids model_dump overhead
            final_messages = [
                {"role": m.role, "content": m.content}
                for m in req.messages if m.content
            ]

            response = await client.chat.completions.create(
                model=DEPLOYMENT,
                messages=final_messages,
                 # Sounds more natural for voice
                stream=True,
                max_completion_tokens=req.max_completion_tokens,
            )

            async for chunk in response:
                # Skip Azure's empty content-filter chunks
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not delta.content:
                    continue

                token = delta.content
                collected.append(token)
                sentence_buffer += token

                # Voice optimization: flush at sentence boundaries
                # so the TTS pipeline receives complete phrases sooner
                if any(p in sentence_buffer for p in FLUSH_CHARS):
                    yield b"data: " + orjson.dumps(
                        chunk.model_dump(exclude_none=True)
                    ) + b"\n\n"
                    sentence_buffer = ""
                else:
                    yield b"data: " + orjson.dumps(
                        chunk.model_dump(exclude_none=True)
                    ) + b"\n\n"

            # Flush any remaining buffer content
            if sentence_buffer and collected:
                pass  # Already yielded token-by-token above

            # Cache the full response for future hits
            if collected:
                RESPONSE_CACHE[ckey] = "".join(collected)

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",   # Critical for Railway/Vercel/Nginx
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# -----------------------------
# 5. Health Check
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(RESPONSE_CACHE)}


# -----------------------------
# 6. Entry Point
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    import os
    import sys

    # Railway provides the PORT environment variable. 
    # We MUST use it, and we MUST bind to 0.0.0.0
    port = int(os.getenv("PORT", 8000))
    
    # Use uvloop for maximum networking performance on Linux (Railway)
    loop_type = "uvloop" if sys.platform != "win32" else "asyncio"

    uvicorn.run(
        "main:app",            # Use the string "file_name:app_variable"
        host="0.0.0.0",        # Mandatory for Railway
        port=port,             # Use the dynamic port from Railway
        loop=loop_type,
        log_level="warning",
        proxy_headers=True,    # Important for ElevenLabs/Railway proxy
        forwarded_allow_ips="*" # Ensures headers like x-api-key pass through
    )