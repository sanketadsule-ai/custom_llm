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
logger = logging.getLogger("maya-turbo-bridge")

# Azure OpenAI Client
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")

# isolated session cache (20-minute expiry)
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
# 3. Turbo Latency Killers
# -----------------------------
async def execute_tool_silently(tool_calls: List[dict]):
    """
    Fire-and-forget execution. 
    Maya keeps talking while the database/n8n updates in the background.
    """
    for tool_call in tool_calls:
        name = tool_call.get("function", {}).get("name")
        args = tool_call.get("function", {}).get("arguments")
        logger.info(f"⚡ Background Exec: {name} with args {args}")
        # Add your n8n webhook call here:
        # await n8n_client.post("/webhook", json={"tool": name, "data": args})

# -----------------------------
# 4. Main Execution Engine
# -----------------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # SESSION ISOLATION: Key = UserID + Last Msg + Msg Count
        msg_history = req.messages
        msg_count = len(msg_history)
        user_id = request.headers.get("x-user-id", "default_donor") 
        last_user_msg = "".join([m.content for m in msg_history if m.role == "user"][-1:])
        
        ckey = hashlib.md5(f"{user_id}:{last_user_msg}:{msg_count}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            logger.info("🚀 Cache Hit: Serving isolated response")
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        try:
            # TURBO KWARGS: Minimal penalties for faster inference
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in msg_history[-12:]],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 80, # Keep Maya's turns short and fast
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "stream_options": {"include_usage": True},
            }
            if req.tools:
                kwargs["tools"] = req.tools
                kwargs["tool_choice"] = "auto"

            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=15.0 
            )

            first_chunk = True
            async for chunk in response:
                if not chunk.choices: continue
                
                delta = chunk.choices[0].delta
                
                # PARALLEL EXECUTION: Trigger tools in background, keep streaming text
                if delta.tool_calls:
                    asyncio.create_task(execute_tool_silently([tc.model_dump() for tc in delta.tool_calls]))
                    continue # Skip sending tool JSON to the audio engine

                if delta.content:
                    collected.append(delta.content)
                    # Yield content immediately to the donor
                    yield b"data: " + orjson.dumps(chunk.model_dump()) + b"\n\n"

                if first_chunk:
                    first_chunk = False
                    await asyncio.sleep(0) # Flush TCP buffer

            if collected:
                RESPONSE_CACHE[ckey] = "".join(collected)

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ Streaming Error: {e}")
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