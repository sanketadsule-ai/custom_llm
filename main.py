import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import AzureOpenAI
from fastapi.responses import JSONResponse

# -------------------
# Azure OpenAI Client
# -------------------
client = AzureOpenAI(
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
# Request Models
# -------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]

# -------------------
# Health Check
# -------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -------------------
# ElevenLabs Endpoint
# -------------------
@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[m.model_dump() for m in req.messages],
            temperature=0.7,
        )

        assistant_text = response.choices[0].message.content or ""

        return JSONResponse(
    status_code=200,
    content={
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": DEPLOYMENT,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_text
            },
            "finish_reason": "stop"
        }]
    }
)

    except Exception as e:
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
