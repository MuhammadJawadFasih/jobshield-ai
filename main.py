from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import io
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------- Basic routes ----------
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
    return FileResponse("static/index.html")


@app.get("/reviewer")
def reviewer_page():
    return FileResponse("static/reviewer.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/test-ai")
def test_ai():
    return {"reply": ask_groq("Say hello in one short sentence.")}


# ---------- Rule-based checker ----------

RULES = [
    ("free", r"\b(registration|processing|visa|medical|training|security|application|booking|agent|service)\s+(fee|fees|charge|charges|deposit)\b|western union|moneygram|bitcoin|crypto|send (us )?money|advance payment|pay (first|in advance|upfront)",
     "Requests an upfront payment or fee.", True),

    ("free_email", r"@(gmail|yahoo|hotmail|outlook)\.",
     "Uses a free personal email instead of a company email.", False),

    ("no_interview", r"no interview|without interview|interview not required",
     "Promises a job without an interview.", False),

    ("whatsapp", r"\bwhatsapp\b",
     "Asks to contact only through WhatsApp.", False),

    ("passport", r"\bpassport\b",
     "Mentions or requests passport details; verify why they are needed.", True),

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


# ---------- AI analysis ----------
SYSTEM_PROMPT = """You are an assistant that checks overseas job offers for signs of fraud.
You do NOT have internet access. Never claim a company, agency or license is verified or real.
Never say "this is a scam" or "this is legit". Use "high risk" or "low risk signals found".

Reply with JSON only, no other text, in exactly this shape:
{
  "risk": "low" or "medium" or "high",
  "summary": "2-3 simple sentences",
  "red_flags": [{"quote": "exact words copied from the offer", "reason": "why it is suspicious"}],
  "checklist": ["things the person should check or ask before accepting"],
  "urgent": true or false
}

Rules:
- Each quote must be copied word for word from the offer. Never invent quotes.
- Write summary, reasons and checklist in the requested language. Keep quotes in their original language.
- Checklist items are things the HUMAN does, for example: check the company with the official government overseas employment office, search the company registration, check the recruiter's license, never pay fees before a verified contract.
- Set urgent to true if the offer asks for money, a passport, or a decision today.
- If the offer has no red flags, return an empty red_flags list and risk "low"."""


def ask_groq_json(offer_text: str, language: str) -> dict:
    response = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        json={
            "model": os.getenv("GROQ_TEXT_MODEL"),
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Language for the explanation: {language}\n\nJob offer:\n{offer_text}"},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start:end + 1])


DEFAULT_CHECKLIST = [
    "Check the company with the official government overseas employment office.",
    "Search for the company's registration and a real office address.",
    "Check the recruiter's license number with the authorities.",
    "Never pay any fee before you have a verified contract.",
]


def analyze_offer(text: str, language: str):
    rules = rule_checks(text)
    ai = None
    try:
        ai = ask_groq_json(text, language)
    except Exception as e:
        print("AI failed:", e)

    flags = list(rules["flags"])
    seen = {(f["quote"], f["reason"]) for f in flags}
    if ai:
        for f in ai.get("red_flags", []):
            q, r = f.get("quote", ""), f.get("reason", "")
            overlaps = any(q in x["quote"] or x["quote"] in q for x in flags)
            if q and q in text and (q, r) not in seen and not overlaps:
                flags.append({"quote": q, "reason": r, "rule": "ai"})
                seen.add((q, r))

    ai_risk = ai.get("risk", "medium") if ai else ("high" if rules["urgent"] else "medium")
    if ai_risk not in ("low", "medium", "high"):
        ai_risk = "medium"
    risk = "high" if rules["urgent"] else ai_risk
    urgent = rules["urgent"] or bool(ai and ai.get("urgent"))

    return {
        "risk": risk,
        "ai_risk": ai_risk,
        "summary": ai.get("summary", "") if ai else "The AI could not run. Showing automatic checks only. A human must review this.",
        "red_flags": flags,
        "checklist": (ai.get("checklist") if ai else None) or DEFAULT_CHECKLIST,
        "urgent": urgent,
        "ai_failed": ai is None,
    }


class AnalyzeIn(BaseModel):
    text: str
    language: str = "English"


@app.post("/analyze-test")
def analyze_test(offer: AnalyzeIn):
    return analyze_offer(offer.text, offer.language)


# ---------- Database ----------
DB_PATH = os.getenv("DB_PATH", "jobshield.db")


def run(sql, params=(), fetch=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        return cur.lastrowid
    finally:
        conn.close()


run("""CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    status TEXT,
    language TEXT,
    offer_text TEXT,
    file_name TEXT,
    urgent INTEGER,
    ai_json TEXT,
    final_json TEXT
)""")


def check_passcode(x_passcode):
    expected = os.getenv("REVIEWER_PASSCODE")
    if not expected or x_passcode != expected:
        raise HTTPException(status_code=401, detail="Wrong passcode")


# ---------- File handling ----------
MAX_FILE = 10 * 1024 * 1024


def extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages[:10]]
        return "\n".join(pages).strip()
    except Exception:
        return ""


# ---------- Person endpoints ----------
@app.post("/api/submit")
def submit(
    text: str = Form(""),
    language: str = Form("English"),
    file: UploadFile = File(None),
):
    text = text.strip()
    file_name = ""
    if file is not None and file.filename:
        data = file.file.read()
        if len(data) > MAX_FILE:
            raise HTTPException(status_code=400, detail="File is too large (max 10 MB).")
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported for now.")
        extracted = extract_pdf_text(data)
        if len(extracted) < 50:
            raise HTTPException(status_code=400, detail="This PDF has no readable text.")
        text = (text + "\n" + extracted).strip()
        file_name = file.filename
    if not text:
        raise HTTPException(status_code=400, detail="Please paste a job offer or upload a file.")
    text = text[:8000]
    analysis = analyze_offer(text, language)
    case_id = run(
        "INSERT INTO cases (created_at, status, language, offer_text, file_name, urgent, ai_json) "
        "VALUES (?, 'pending', ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), language, text, file_name,
         1 if analysis["urgent"] else 0, json.dumps(analysis)),
    )
    return {"id": case_id, "status": "pending"}


@app.get("/api/case/{case_id}")
def case_status(case_id: int):
    row = run("SELECT status, final_json FROM cases WHERE id = ?", (case_id,), "one")
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    if row["status"] == "pending":
        return {"status": "pending"}
    final = json.loads(row["final_json"])
    if row["status"] == "needs_info":
        return {"status": "needs_info", "note": final["note"]}
    return {"status": "approved", **final}


# ---------- Reviewer endpoints ----------
@app.get("/api/reviewer/cases")
def reviewer_cases(x_passcode: str = Header(default=None)):
    check_passcode(x_passcode)
    rows = run(
        "SELECT id, created_at, status, urgent, ai_json FROM cases "
        "ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, urgent DESC, id DESC",
        fetch="all",
    )
    return [
        {"id": r["id"], "created_at": r["created_at"], "status": r["status"],
         "urgent": bool(r["urgent"]), "ai_risk": json.loads(r["ai_json"])["risk"]}
        for r in rows
    ]


@app.get("/api/reviewer/case/{case_id}")
def reviewer_case(case_id: int, x_passcode: str = Header(default=None)):
    check_passcode(x_passcode)
    row = run("SELECT * FROM cases WHERE id = ?", (case_id,), "one")
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    return {
        "id": row["id"],
        "status": row["status"],
        "language": row["language"],
        "offer_text": row["offer_text"],
        "file_name": row["file_name"],
        "analysis": json.loads(row["ai_json"]),
        "final": json.loads(row["final_json"]) if row["final_json"] else None,
    }


class DecideIn(BaseModel):
    decision: str      # "approved" or "needs_info"
    final_risk: str    # "low", "medium" or "high"
    summary: str
    note: str = ""


@app.post("/api/reviewer/case/{case_id}/decide")
def reviewer_decide(case_id: int, body: DecideIn, x_passcode: str = Header(default=None)):
    check_passcode(x_passcode)
    if body.decision not in ("approved", "needs_info"):
        raise HTTPException(status_code=400, detail="decision must be approved or needs_info")
    if body.final_risk not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="final_risk must be low, medium or high")
    row = run("SELECT ai_json FROM cases WHERE id = ?", (case_id,), "one")
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    ai = json.loads(row["ai_json"])
    final = {
        "final_risk": body.final_risk,
        "ai_risk": ai["risk"],
        "changed": body.final_risk != ai["risk"],
        "summary": body.summary,
        "note": body.note,
        "red_flags": ai["red_flags"],
        "checklist": ai["checklist"],
    }
    run("UPDATE cases SET status = ?, final_json = ? WHERE id = ?",
        (body.decision, json.dumps(final), case_id))
    return {"ok": True}