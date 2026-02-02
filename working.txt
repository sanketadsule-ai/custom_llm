import os
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import AsyncAzureOpenAI
from fastapi.responses import StreamingResponse

# -------------------
# Azure OpenAI Client (Async for Streaming)
# -------------------
# Make sure these environment variables are set in Railway
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# -------------------
# FastAPI App
# -------------------
app = FastAPI()

# -------------------
# Request Models (OpenAI Spec)
# -------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    stream: Optional[bool] = True  # ElevenLabs usually sends this as True

# -------------------
# Health Check
# -------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "Server is running"}

# -------------------
# ElevenLabs Endpoint
# -------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest):
    async def event_generator():
        try:
            # Create the stream from Azure
            response = await client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[m.model_dump() for m in req.messages],
                temperature=0.7,
                stream=True
            )

            async for chunk in response:
                # Convert the chunk object to a dictionary
                chunk_dict = chunk.model_dump()
                
                # Format as Server-Sent Events (SSE)
                # Each chunk must start with "data: " and end with "\n\n"
                yield f"data: {json.dumps(chunk_dict)}\n\n"

            # ElevenLabs requires the [DONE] signal to stop synthesis
            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"🔥 Error during stream: {str(e)}")
            # If it fails, we still send [DONE] so the connection doesn't hang
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Railway provides the PORT environment variable automatically
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)