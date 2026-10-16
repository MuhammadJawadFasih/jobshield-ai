import re
from pydantic import BaseModel
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
RULES = [
    ("fee", r"\b(fee|deposit|payment|western union|bitcoin|crypto|send money|pay a|pay the)\b",
     "Asks for money upfront. Real employers do not charge job seekers.", True),
    ("free_email", r"@(gmail|yahoo|hotmail|outlook)\.",
     "Uses a free personal email instead of a company email.", False),
    ("no_interview", r"no interview|without interview|interview not required",
     "Promises a job without an interview.", False),
    ("whatsapp", r"\bwhatsapp\b",
     "Asks to contact only through WhatsApp.", False),
    ("passport", r"\bpassport\b",
     "Asks for passport details.", True),
    ("pressure", r"today only|urgent|immediately|last chance|limited slots|within 24 hours",
     "Pressures you to decide quickly.", False),
]


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def rule_checks(text: str):
    flags = []
    urgent = False
    for sentence in split_sentences(text):
        for name, pattern, reason, is_urgent in RULES:
            if re.search(pattern, sentence, re.IGNORECASE):
                flags.append({"quote": sentence, "reason": reason, "rule": name})
                if is_urgent:
                    urgent = True
    return {"flags": flags, "urgent": urgent}


class OfferIn(BaseModel):
    text: str


@app.post("/rules-test")
def rules_test(offer: OfferIn):
    return rule_checks(offer.text)