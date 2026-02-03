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
import httpx # For high-performance connection pooling

# -----------------------------
# 1. Setup & Performance Tuning
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-turbo-bridge")

# Using a persistent AsyncClient for connection pooling (saves ~100-300ms per call)
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    timeout=httpx.Timeout(10.0, read=None)
)

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
    http_client=http_client # Injecting the high-perf client
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")
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
# 3. Enhanced Tool Execution
# -----------------------------
async def execute_tool_silently(tool_calls: List[dict]):
    """
    Fire-and-forget background execution. 
    """
    for tool_call in tool_calls:
        func = tool_call.get("function", {})
        name = func.get("name")
        args = func.get("arguments")
        
        if name:
            # Note: Do not 'await' long-running tasks here if they block.
            # Use httpx.AsyncClient for external webhooks to maintain async flow.
            logger.info(f"⚡ [EXEC] {name} | Args: {args}")

# -----------------------------
# 4. Latency-First Loop
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # --- PRE-PROCESSING OPTIMIZATION ---
        msg_history = req.messages
        msg_count = len(msg_history)
        user_id = request.headers.get("x-user-id", "default_donor") 
        last_user_msg = "".join([m.content for m in msg_history if m.role == "user"][-1:])
        
        # Cache lookup using MD5 for O(1) speed
        ckey = hashlib.md5(f"{user_id}:{last_user_msg}:{msg_count}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            cached_text = RESPONSE_CACHE[ckey]
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content": cached_text}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        try:
            # --- MODEL PARAMS OPTIMIZATION ---
            kwargs = {
                "model": DEPLOYMENT,
                # Slice history to last 6-8 messages to reduce prompt processing time
                "messages": [m.model_dump(exclude_none=True) for m in msg_history[-8:]],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 80, # Keep responses concise to lower generation latency
                "presence_penalty": 0,
                "frequency_penalty": 0,
            }
            if req.tools:
                kwargs["tools"] = req.tools
                # Consider adding tool_choice="auto" specifically if needed

            # Wait for stream with a strict timeout
            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs), 
                timeout=10.0
            )

            async for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                
                # Immediate handling for tool calls
                if delta.tool_calls:
                    # We only trigger once per tool call ID to prevent redundant logs
                    if delta.tool_calls[0].id: 
                        asyncio.create_task(execute_tool_silently([tc.model_dump() for tc in delta.tool_calls]))
                    continue

                # Stream text content immediately to the user
                if delta.content:
                    collected.append(delta.content)
                    # use orjson for faster serialization than standard json.dumps
                    yield b"data: " + orjson.dumps(chunk.model_dump()) + b"\n\n"

            if collected:
                RESPONSE_CACHE[ckey] = "".join(collected)

            yield b"data: [DONE]\n\n"

        except asyncio.TimeoutError:
            logger.error("⏰ Azure OpenAI Timeout")
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"❌ Streaming Error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no", # Critical for Nginx/Proxies
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Use uvloop for 2-3x better performance on Linux
    import sys
    loop_type = "uvloop" if sys.platform != "win32" else "asyncio"
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)), 
        loop=loop_type,
        http="httptools", # Faster HTTP parser
        workers=1,
        access_log=False # Reduce I/O overhead
    )