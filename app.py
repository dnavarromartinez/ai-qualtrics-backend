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
# APP
# -----------------------
app = FastAPI()

# -----------------------
# CORS (FIXED FOR QUALTRICS)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # simplest + most robust for Qualtrics
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# OPENAI
# -----------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# LOG FILE (Render persistent disk)
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
# SIMPLE MEMORY (per participant)
# -----------------------
conversation_memory = {}

# -----------------------
# REQUEST MODEL
# -----------------------
class ChatRequest(BaseModel):
    participant_id: str
    message: str
    condition: str | None = None

# -----------------------
# CHAT ENDPOINT
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):

    pid = req.participant_id
    timestamp = datetime.utcnow().isoformat()

    # -----------------------
    # INIT MEMORY
    # -----------------------
    if pid not in conversation_memory:
        conversation_memory[pid] = [
            {
                "role": "system",
                "content": "You are a neutral assistant helping users make decisions."
            }
        ]

    # -----------------------
    # ADD USER MESSAGE
    # -----------------------
    conversation_memory[pid].append({
        "role": "user",
        "content": req.message
    })

    # -----------------------
    # CALL OPENAI WITH MEMORY
    # -----------------------
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=conversation_memory[pid],
        temperature=0.7
    )

    ai_reply = response.choices[0].message.content

    # -----------------------
    # ADD ASSISTANT MESSAGE
    # -----------------------
    conversation_memory[pid].append({
        "role": "assistant",
        "content": ai_reply
    })

    # -----------------------
    # LOG TO CSV
    # -----------------------
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            pid,
            req.condition,
            "user",
            req.message
        ])

        writer.writerow([
            timestamp,
            pid,
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