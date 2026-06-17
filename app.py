from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import csv
import os

# -----------------------
# App
# -----------------------
app = FastAPI()

# -----------------------
# CORS (required for Qualtrics)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for experiment; restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# OpenAI client
# -----------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# Persistent log file (Render disk)
# -----------------------
LOG_FILE = "/data/chat_logs.csv"

# Create file + header if it doesn't exist
if not Path(LOG_FILE).exists():
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "participant_id",
            "condition",
            "role",
            "content"
        ])

# -----------------------
# Request schema
# -----------------------
class ChatRequest(BaseModel):
    participant_id: str
    message: str
    condition: str | None = None

# -----------------------
# Endpoint
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a neutral assistant helping users make decisions."
            },
            {
                "role": "user",
                "content": req.message
            }
        ],
        temperature=0.7
    )

    ai_reply = response.choices[0].message.content
    timestamp = datetime.utcnow().isoformat()

    # -----------------------
    # LOG USER + ASSISTANT MESSAGE
    # -----------------------
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # user message
        writer.writerow([
            timestamp,
            req.participant_id,
            req.condition,
            "user",
            req.message
        ])

        # assistant message
        writer.writerow([
            timestamp,
            req.participant_id,
            req.condition,
            "assistant",
            ai_reply
        ])

    # -----------------------
    # RESPONSE TO QUALTRICS
    # -----------------------
    return {
        "reply": ai_reply
    }