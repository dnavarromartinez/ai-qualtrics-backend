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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# OPENAI
# -----------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# LOG FILE (persistent disk)
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

    timestamp = datetime.utcnow().isoformat()

    # -----------------------
    # OPENAI CALL (stateless chat)
    # -----------------------
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

    # -----------------------
    # LOGGING (simple append-only)
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