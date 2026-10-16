import os
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def ask_groq(prompt: str) -> str:
    response = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        json={
            "model": os.getenv("GROQ_TEXT_MODEL"),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@app.get("/")
def home():
    return {"message": "Hello JobShield"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/test-ai")
def test_ai():
    return {"reply": ask_groq("Say hello in one short sentence.")}