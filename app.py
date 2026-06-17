from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import csv
import os
import pandas as pd

# -----------------------
# App
# -----------------------
app = FastAPI()

# -----------------------
# CORS (Qualtrics)
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
# Log file (persistent disk)
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
            "content",
            "msg_index"
        ])

# -----------------------
# Request schema
# -----------------------
class ChatRequest(BaseModel):
    participant_id: str
    message: str
    condition: str | None = None


# -----------------------
# PERSISTENT INDEX (derived from logs)
# -----------------------
def get_next_msg_index(participant_id: str) -> int:
    if not Path(LOG_FILE).exists():
        return 1

    df = pd.read_csv(LOG_FILE)

    p_df = df[df["participant_id"] == participant_id]

    if p_df.empty:
        return 1

    # count USER messages only (turn-based index)
    return len(p_df[p_df["role"] == "user"]) + 1


# -----------------------
# Chat endpoint
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):

    # -----------------------
    # get turn index (PERSISTENT)
    # -----------------------
    msg_index = get_next_msg_index(req.participant_id)

    # -----------------------
    # build conversation (no memory needed here for logging)
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
    timestamp = datetime.utcnow().isoformat()

    # -----------------------
    # LOG INTERACTION
    # -----------------------
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            req.participant_id,
            req.condition,
            "user",
            req.message,
            msg_index
        ])

        writer.writerow([
            timestamp,
            req.participant_id,
            req.condition,
            "assistant",
            ai_reply,
            msg_index
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