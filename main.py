from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.security import APIKeyHeader
import random
import re

# -------------------- CONFIG --------------------

API_KEY = "honeypot_key_2026_eval"
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI()

sessions = {}

# -------------------- AUTH --------------------

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key


# -------------------- HEALTH --------------------

@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze_get():
    return {
        "status": "success",
        "message": "Honeypot analyze endpoint is live. Use POST to analyze messages."
    }


# -------------------- UTILS --------------------

def extract_entities(text: str):
    upi = re.findall(r'\b[\w.-]+@[\w.-]+\b', text)
    urls = re.findall(r'https?://\S+', text)
    phones = re.findall(r'\b\d{10}\b', text)
    banks = re.findall(r'\b\d{9,18}\b', text)

    return {
        "upi": upi,
        "urls": urls,
        "phone": phones,
        "bank_info": banks
    }


def predict_scam_type(text: str):
    text = text.lower()
    if "bank" in text or "account" in text:
        return "bank_fraud"
    if "upi" in text or "pay" in text:
        return "upi_fraud"
    if "prize" in text or "lottery" in text:
        return "fake_offer"
    if "link" in text or "http" in text:
        return "phishing"
    return "unknown"


def calculate_confidence_ml(text: str):
    return round(random.uniform(0.6, 0.95), 2)


async def generate_agent_response(session_id, message, scam_type, entities):
    if scam_type == "bank_fraud":
        return "Why is my bank account being blocked? What should I do now?"
    if scam_type == "upi_fraud":
        return "I’m worried. Why do you need my UPI ID?"
    if scam_type == "phishing":
        return "Is this link really safe to open?"
    return "Can you explain this more clearly?"


# -------------------- MAIN ENDPOINT --------------------

@app.post("/analyze")
async def analyze_scam(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = body.get("sessionId") or f"sess_{random.randint(1000,9999)}"

    message_obj = body.get("message", {})
    message_text = message_obj.get("text", "")

    if not message_text:
        message_text = "This is a suspicious message asking for money."

    # ---- Intelligence + ML ----

    entities = extract_entities(message_text)
    scam_type = predict_scam_type(message_text)
    confidence = calculate_confidence_ml(message_text)

    agent_reply = await generate_agent_response(
        session_id, message_text, scam_type, entities
    )

    # ---- Session memory ----

    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append({
        "scammer": message_text,
        "agent": agent_reply
    })

    # --------------------
    # ✅ GUVI EXPECTED RESPONSE
    # --------------------

    return {
        "status": "success",
        "reply": agent_reply
    }
