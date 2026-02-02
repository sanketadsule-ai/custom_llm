import os
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Any
from openai import AsyncAzureOpenAI
from fastapi.responses import StreamingResponse

# -------------------
# Azure OpenAI Client
# -------------------
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

app = FastAPI()

# -------------------
# Updated Models for Tool Support
# -------------------
class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None # Allow receiving tool calls in history

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    tools: Optional[List[Any]] = None      # ElevenLabs sends tools here
    stream: Optional[bool] = True

@app.get("/")
def health():
    return {"status": "ok", "message": "Maya Agent Server Running"}

@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest):
    async def event_generator():
        try:
            # Prepare arguments for Azure
            # We pass 'tools' directly to Azure so the LLM knows what it can do
            kwargs = {
                "model": DEPLOYMENT,
                "messages": [m.model_dump(exclude_none=True) for m in req.messages],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 150,
                "presence_penalty":0, # Higher values increase calculation time
                "frequency_penalty":0
            }
            
            if req.tools:
                kwargs["tools"] = req.tools
                kwargs["tool_choice"] = "auto"

            response = await client.chat.completions.create(**kwargs)

            async for chunk in response:
                chunk_dict = chunk.model_dump()
                # Ensure we yield the chunk so ElevenLabs sees tool_calls or content
                yield f"data: {json.dumps(chunk_dict)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"🔥 Error: {str(e)}")
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)