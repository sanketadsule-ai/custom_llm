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
logger = logging.getLogger("maya-ultra-low-latency")

limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
http_client = httpx.AsyncClient(limits=limits, timeout=60.0)

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    http_client=http_client
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

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
    max_tokens: int = 100

# -----------------------------
# 3. Optimized Logic
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        user_id = request.headers.get("x-user-id", "anonymous")
        user_context = "".join([m.content for m in req.messages if m.role == "user"][-10:])
        ckey = hashlib.md5(f"{user_id}:{user_context}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        try:
            # --- CONTEXT PERSISTENCE LOGIC ---
            # Extract the system message (which contains the rules and variables)
            system_msg = next((m for m in req.messages if m.role == "system"), None)
            
            # Get only the most recent conversation history (last 9 messages)
            # This prevents the context from becoming too large/expensive
            history = [m for m in req.messages if m.role != "system"][-9:]
            
            final_messages = []
            if system_msg:
                # RE-INJECT VARIABLES (Optional: Replace strings if passed in headers)
                # content = system_msg.content.replace("{{customer_name}}", request.headers.get("x-customer-name", "Donor"))
                final_messages.append(system_msg.model_dump(exclude_none=True))
            
            final_messages.extend([m.model_dump(exclude_none=True) for m in history])

            kwargs = {
                "model": DEPLOYMENT,
                "messages": final_messages,
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

            first_chunk = True
            async for chunk in response:
                chunk_data = chunk.model_dump(exclude_none=True)
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

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

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