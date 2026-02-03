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
RESPONSE_CACHE = LRUCache(maxsize=500)

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Any]
    tools: Optional[List[Any]] = None
    stream: bool = True
    max_tokens: int = 150

async def execute_tool_background(tool_list: List[dict]):
    """This function is now wrapped to ensure logs only happen once."""
    for tool in tool_list:
        logger.info(f"⚡ [SINGLE EXEC] {tool.get('name')} | Args: {tool.get('args')}")

@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        # --- REQUEST SCOPE STATE ---
        # These variables reset every time a new call comes in
        executed_in_this_turn = False 
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
            }
            if req.tools:
                kwargs["tools"] = req.tools
                kwargs["parallel_tool_calls"] = False # Force 1 tool at a time

            response = await client.chat.completions.create(**kwargs)

            async for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # 1. Capture Tool Data
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_buffer:
                            tool_buffer[idx] = {"name": "", "args": ""}
                        if tc.function.name:
                            tool_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_buffer[idx]["args"] += tc.function.arguments

                # 2. THE LOCK: Only fire if finish_reason is exactly 'tool_calls'
                # AND we haven't flipped our local toggle yet.
                if finish_reason == "tool_calls" and not executed_in_this_turn:
                    if tool_buffer:
                        # Fire and forget
                        asyncio.create_task(execute_tool_background(list(tool_buffer.values())))
                        executed_in_this_turn = True 

                # 3. Stream Speech
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
    # STRICTLY 1 WORKER to prevent multi-process log duplication
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, access_log=False)