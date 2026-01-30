import os
import openai
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load Azure API key from environment (Railway Variables)
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
if not AZURE_KEY:
    raise ValueError("AZURE_OPENAI_KEY is not set!")

# Azure OpenAI config
openai.api_type = "azure"
openai.api_base = "https://impactguru-openai.openai.azure.com/"
openai.api_version = "2025-01-01-preview"
openai.api_key = AZURE_KEY

DEPLOYMENT_NAME = "gpt-4o"  # Must match your Azure deployment name exactly

class PromptRequest(BaseModel):
    prompt: str

@app.post("/custom-llm")
async def custom_llm(req: PromptRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,  # Azure deployment
            messages=[{"role": "user", "content": req.prompt}]
        )
        return {"output": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure OpenAI call failed: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
