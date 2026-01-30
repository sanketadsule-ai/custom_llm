
import os
from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai



app = FastAPI()

# Load your Azure API key from Railway environment
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
if not AZURE_KEY:
    raise ValueError("AZURE_OPENAI_KEY is not set!")

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_base = "https://impactguru-openai.openai.azure.com/"
openai.api_version = "2025-01-01-preview"
openai.api_key = AZURE_KEY

DEPLOYMENT_NAME = "gpt-4o"  # This must match your Azure deployment exactly

@app.post("/custom-llm")
async def custom_llm(payload: dict):
    prompt = payload.get("prompt", "")
    if not prompt:
        return {"output": "No prompt provided."}
    
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,  # Azure deployment name
            messages=[{"role": "user", "content": prompt}]
        )
        return {"output": response.choices[0].message.content}
    except Exception as e:
        return {"output": f"Error calling Azure OpenAI: {e}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
