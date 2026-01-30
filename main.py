
import os
from dotenv import load_dotenv
load_dotenv

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

# -------- Azure OpenAI Configuration --------
openai.api_type = "azure"
openai.api_base = "https://impactguru-openai.openai.azure.com/"
openai.api_version = "2023-12-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")  # set in Railway

# -------- FastAPI App --------
app = FastAPI()

class ElevenLLMRequest(BaseModel):
    prompt: str

@app.post("/custom-llm")
def custom_llm(req: ElevenLLMRequest):
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a calm, friendly phone assistant. Speak briefly and naturally."
                },
                {
                    "role": "user",
                    "content": req.prompt
                }
            ],
            temperature=0.2,
            max_tokens=300,
        )

        text_output = response["choices"][0]["message"]["content"]
        return {"output": text_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
