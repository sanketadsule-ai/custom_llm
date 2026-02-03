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
import httpx

# 1. SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maya-turbo-bridge")

http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    timeout=httpx.Timeout(10.0, read=None)
)

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview", 
    http_client=http_client
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AUTH_KEY = os.getenv("CUSTOM_LLM_API_KEY")
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=1200)

app = FastAPI()

# 2. MODELS
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

# 3. TOOL EXECUTION
async def execute_tool_silently(tool_name: str, args: str):
    """
    Fire-and-forget background execution. 
    Maya is already speaking while this runs.
    """
    logger.info(f"⚡ [SINGLE EXEC] {tool_name} | Args: {args}")
    # ADD YOUR Webhook/n8n logic here

# 4. THE LOOP
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if request.headers.get("x-api-key") != AUTH_KEY:
        raise HTTPException(status_code=401)

    async def event_generator():
        msg_history = req.messages
        msg_count = len(msg_history)
        user_id = request.headers.get("x-user-id", "default_donor") 
        last_user_msg = "".join([m.content for m in msg_history if m.role == "user"][-1:])
        ckey = hashlib.md5(f"{user_id}:{last_user_msg}:{msg_count}".encode()).hexdigest()
        
        if ckey in RESPONSE_CACHE:
            yield b"data: " + orjson.dumps({"choices":[{"delta":{"content":RESPONSE_CACHE[ckey]}}]}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        collected = []
        
        # --- THE FIX: PER-REQUEST STATE ---
        executed_tool_ids = set() # Track IDs we already fired
        tool_buffer = {}          # Buffer chunks for each tool index

        try:
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in msg_history[-8:]],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 100,
            }
            if req.tools:
                kwargs["tools"] = req.tools
                # Disabling parallel calls can also help stabilize 3.5-turbo
                kwargs["parallel_tool_calls"] = False 

            response = await client.chat.completions.create(**kwargs)

            async for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                
                # 1. Accumulate Tool Chunks
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_buffer:
                            tool_buffer[idx] = {"id": None, "name": "", "args": ""}
                        
                        if tc.id: tool_buffer[idx]["id"] = tc.id
                        if tc.function.name: tool_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments: tool_buffer[idx]["args"] += tc.function.arguments

                # 2. Trigger Task ONLY on Finish Signal
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason == "tool_calls":
                    for idx, data in tool_buffer.items():
                        t_id = data["id"] or f"idx_{idx}"
                        if t_id not in executed_tool_ids:
                            asyncio.create_task(execute_tool_silently(data["name"], data["args"]))
                            executed_tool_ids.add(t_id)

                # 3. Stream Text
                if delta.content:
                    collected.append(delta.content)
                    yield b"data: " + orjson.dumps(chunk.model_dump()) + b"\n\n"

            if collected:
                RESPONSE_CACHE[ckey] = "".join(collected)
            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), workers=1)