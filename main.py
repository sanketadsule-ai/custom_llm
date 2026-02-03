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

# 1. Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-ultra")

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

# We use two caches: one for text, one for tracking tool execution status
RESPONSE_CACHE = LRUCache(maxsize=500)
TOOL_EXECUTION_TRACKER = LRUCache(maxsize=500) # Track: ckey -> bool

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Any]
    tools: Optional[List[Any]] = None
    stream: bool = True
    max_tokens: int = 150

async def execute_tool_background(tool_list: List[dict], ckey: str):
    """Execution with a Global Lock check."""
    # DOUBLE CHECK: If another worker/thread already finished this ckey, abort.
    if TOOL_EXECUTION_TRACKER.get(ckey) == "EXECUTED":
        return

    TOOL_EXECUTION_TRACKER[ckey] = "EXECUTED"
    for tool in tool_list:
        logger.info(f"⚡ [SINGLE EXEC] {tool.get('name')} | Turn: {ckey[:8]}")
        # Your n8n/webhook logic here

@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    # Generate a unique key for THIS turn (User ID + Last Msg Content)
    user_id = request.headers.get("x-user-id", "anon")
    last_msg = str(req.messages[-1])
    ckey = hashlib.md5(f"{user_id}:{last_msg}".encode()).hexdigest()

    async def event_generator():
        # Local lock for this specific stream instance
        stream_instance_fired = False 
        tool_buffer = {}
        collected_text = []

        try:
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m if isinstance(m, dict) else m.model_dump(exclude_none=True) for m in req.messages[-10:]],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": req.max_tokens,
                "stream_options": {"include_usage": True},
                "parallel_tool_calls": False
            }
            if req.tools:
                kwargs["tools"] = req.tools

            response = await client.chat.completions.create(**kwargs)

            async for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # 1. Reconstruct Tool JSON
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_buffer:
                            tool_buffer[idx] = {"name": "", "args": ""}
                        if tc.function.name: tool_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments: tool_buffer[idx]["args"] += tc.function.arguments

                # 2. TRIGGER WITH TRIPLE LOCK
                # Check 1: Finish reason hit?
                # Check 2: Has this specific stream fired yet?
                # Check 3: Has this conversation turn (ckey) been marked as executed in global cache?
                if finish_reason == "tool_calls" and not stream_instance_fired:
                    if ckey not in TOOL_EXECUTION_TRACKER:
                        if tool_buffer:
                            asyncio.create_task(execute_tool_background(list(tool_buffer.values()), ckey))
                            stream_instance_fired = True

                if delta.content:
                    collected_text.append(delta.content)
                    yield b"data: " + orjson.dumps(chunk.model_dump(exclude_none=True)) + b"\n\n"

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Set workers to 1 to ensure the TOOL_EXECUTION_TRACKER (memory) is shared
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, access_log=False)