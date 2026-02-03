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

# Use TTLCache to auto-clear memory every 20 mins per session
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=1200)

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
# 3. Parallel Tool Dispatch
# -----------------------------
async def execute_tool_background(tool_call):
    """Executes tools in parallel so the agent keeps talking without a 1.2s lag."""
    logger.info(f"Background Tool Dispatch: {tool_call.get('function', {}).get('name')}")
    # Integration logic for n8n/DB goes here

# -----------------------------
# 4. Optimized Logic
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # THE FIX: STATE-AWARE CACHE KEY
        # We include the message count. "yes" at message 4 is different from "yes" at message 6.
        msg_history = req.messages
        msg_count = len(msg_history)
        
        # Pull user identity (assume first message or custom header holds it)
        user_id = request.headers.get("x-user-id", "default_user") 
        last_user_msg = "".join([m.content for m in msg_history if m.role == "user"][-2:])
        
        # Combine ID + Content + Position in conversation to prevent loops
        ckey_raw = f"{user_id}:{last_user_msg}:{msg_count}"
        ckey = hashlib.md5(ckey_raw.encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            logger.info("Cache Hit - Serving isolated response")
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        try:
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in msg_history[-12:]],
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
                
                # Check for tool calls and trigger background execution immediately
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
                    await asyncio.sleep(0) # Flush first token immediately

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