from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# 1. Put your OpenAI key here
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 2. This defines what Qualtrics will send us
class ChatRequest(BaseModel):
    participant_id: str
    message: str
    condition: str | None = None


@app.post("/chat")
def chat(req: ChatRequest):

    # 3. System instruction (you can later change this for experiments)
    system_prompt = """
    You are an assistant helping users think through scenarios.
    Be neutral and do not encourage illegal or unethical behavior.
    """

    # 4. Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        temperature=0.7
    )

    reply = response.choices[0].message.content

    # 5. Send result back to Qualtrics
    return {"reply": reply}