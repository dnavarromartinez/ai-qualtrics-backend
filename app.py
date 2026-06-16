from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# -----------------------
# CORS (REQUIRED for Qualtrics)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for experiment/testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# OpenAI client
# -----------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------
# Request schema
# -----------------------
class ChatRequest(BaseModel):
    participant_id: str
    message: str
    condition: str | None = None


# -----------------------
# Main endpoint
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):

    system_prompt = "You are a neutral assistant helping with decision-making tasks."

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        temperature=0.7
    )

    return {
        "reply": response.choices[0].message.content
    }