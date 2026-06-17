from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
# CORS (Qualtrics compatibility)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# OpenAI client
# -----------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# Conversation memory (NEW)
# -----------------------
conversations = {}

# -----------------------
# Persistent log file (Render disk)
# -----------------------
LOG_FILE = "/data/chat_logs.csv"

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
# Chat endpoint (UPDATED)
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):

    # -----------------------
    # init conversation if new participant
    # -----------------------
    if req.participant_id not in conversations:
        conversations[req.participant_id] = []

    history = conversations[req.participant_id]

    # add user message
    history.append({"role": "user", "content": req.message})

    # call OpenAI with full conversation history
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a neutral assistant helping users make decisions."
            },
            *history
        ],
        temperature=0.7
    )

    ai_reply = response.choices[0].message.content
    timestamp = datetime.utcnow().isoformat()

    # add assistant reply to memory
    history.append({"role": "assistant", "content": ai_reply})

    # -----------------------
    # LOG INTERACTION (persistent)
    # -----------------------
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            req.participant_id,
            req.condition,
            "user",
            req.message
        ])

        writer.writerow([
            timestamp,
            req.participant_id,
            req.condition,
            "assistant",
            ai_reply
        ])

    return {"reply": ai_reply}

# -----------------------
# DOWNLOAD LOGS
# -----------------------
@app.get("/download-logs")
def download_logs():
    return FileResponse(
        LOG_FILE,
        media_type="text/csv",
        filename="chat_logs.csv"
    )