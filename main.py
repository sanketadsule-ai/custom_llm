import os
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# --- Load Azure API key from environment ---
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
if not AZURE_KEY:
    raise ValueError("AZURE_OPENAI_KEY is not set!")

# --- Configure Azure OpenAI ---
openai.api_type = "azure"
openai.api_base = "https://impactguru-openai.openai.azure.com/"
openai.api_version = "2025-01-01-preview"
openai.api_key = AZURE_KEY

# --- Azure deployment name ---
DEPLOYMENT_NAME = "gpt-4o"  # must match your Azure deployment

# --- Request model ---
class PromptRequest(BaseModel):
    prompt: str

# --- Main endpoint ---
@app.post("/custom-llm")
async def custom_llm(req: PromptRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": req.prompt}]
        )
        return {"output": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure OpenAI call failed: {e}")

# --- Alias endpoint for ElevenLabs automatic /chat/completions path ---
@app.post("/custom-llm/chat/completions")
async def custom_llm_alias(req: PromptRequest):
    return await custom_llm(req)

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
