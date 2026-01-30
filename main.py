from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = None
    messages: List[Message]

app = FastAPI()

@app.post("/custom-llm/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        # Take the first user message
        user_prompt = req.messages[0].content if req.messages else ""
        # Call Azure
        response = openai.ChatCompletion.create(
            engine="gpt-4o",
            messages=[{"role": "user", "content": user_prompt}]
        )
        return {"output": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
