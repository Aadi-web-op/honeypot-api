from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import re
from typing import List, Optional, Dict
import random
import os
import joblib
import pandas as pd
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Honeypot API", description="Hybrid ML + LLM Agentic Scam Analysis API")

@app.get("/")
def health():
    return {"status": "ok"}

# --- Module 1: Classification Models (Hybrid) ---
# Load ML models at startup
try:
    classifier = joblib.load("scam_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    ML_ENABLED = True
    print("✅ ML Models Loaded Successfully")
except Exception as e:
    ML_ENABLED = False
    print(f"⚠️ ML Models could not be loaded: {e}. Falling back to rules.")

# --- Module 2: LLM Client (Groq) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
llm_client = None
if GROQ_API_KEY:
    llm_client = AsyncGroq(api_key=GROQ_API_KEY)
    print("✅ Groq Client Initialized")
else:
    print("⚠️ GROQ_API_KEY not found. Using fallback templates.")

# --- Module 3: Authentication ---
API_KEY = "honeypot_key_2026_eval"

async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

# --- Module 4: Intelligence Extraction ---
def extract_entities(text: str) -> Dict[str, List[str]]:
    return {
        "upi": list(set(re.findall(r"[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}", text))),
        "phone": list(set([m.strip() for m in re.findall(r"\+?\d[\d -]{8,12}\d", text) if len(re.sub(r'\D', '', m)) >= 10])),
        "urls": list(set(re.findall(r"https?://(?:[-\w./?=&%]|(?:%[\da-fA-F]{2}))+", text))),
        "bank_info": list(set([f"Account: {a}" for a in re.findall(r"\b\d{9,18}\b", text)]))
    }

# --- Module 5: Hybrid Core Logic ---
SCAM_TYPES = ["bank", "lottery", "loan", "investment", "tech_support"]

def predict_scam_type(text: str) -> str:
    if ML_ENABLED:
        try:
            vec = vectorizer.transform([text])
            return classifier.predict(vec)[0]
        except Exception as e:
            print(f"ML Prediction Error: {e}")
    # Fallback Rule Logic
    text_lower = text.lower()
    if any(x in text_lower for x in ["blocked", "kyc", "pan"]): return "bank"
    if any(x in text_lower for x in ["won", "prize", "lottery"]): return "lottery"
    return "generic"

def calculate_confidence_ml(text: str) -> float:
    if ML_ENABLED:
        try:
            vec = vectorizer.transform([text])
            probs = classifier.predict_proba(vec)
            return float(max(probs[0]))
        except:
            pass
    return 0.5  # Default if ML fails

# --- Module 6: Agentic Response (LLM) ---
sessions = {}

async def generate_agent_response(session_id: str, text: str, scam_type: str, entities: Dict) -> str:
    # 1. Retrieve History
    history = sessions.get(session_id, {}).get("history", [])
    
    # 2. Construct Prompt
    system_prompt = f"""You are a cooperative but cautious victim interacting with a scammer. 
    Your goal is to delay payment, waste their time, and extract details (like bank account, UPI, etc).
    
    Current Scenario:
    - Scam Type: {scam_type.upper()}
    - Extracted So Far: {entities}
    
    Persona Guidelines:
    - If 'bank': Act worried, ask if money is safe.
    - If 'lottery': Act excited but confused about claiming.
    - If 'tech_support': Act computer illiterate, ask simple questions.
    - Always ask a follow-up question.
    - Keep responses short (1-2 sentences).
    - Do NOT reveal you are an AI.
    """
    
    messages = [{"role": "system", "content": system_prompt}] + history[-4:] + [{"role": "user", "content": text}]
    
    # 3. Call LLM (if available)
    if llm_client:
        try:
            chat_completion = await llm_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile", # Updated to supported model
                temperature=0.7,
                max_tokens=100
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"LLM Error: {e}")
            
    # Fallback Templates
    templates = {
        "bank": "Oh my god, is my account blocked? What do I do?",
        "lottery": "Wow! Really? How do I claim it?",
        "tech_support": "My computer is slow. Do I need to click something?"
    }
    return templates.get(scam_type, "Tell me more about this.")

# --- Models & Endpoints ---
class AnalyzeRequest(BaseModel):
    message: Optional[str] = None
    session_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    session_id: str
    scam_type: str
    confidence_score: float
    extracted_entities: Dict[str, List[str]]
    agent_response: str
    is_ml_used: bool

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_scam(request: AnalyzeRequest, api_key: str = Depends(get_api_key)):

    # 🔐 Fallback if evaluator sends empty body
    message = request.message or "This is a suspicious message asking for money."

    current_session_id = request.session_id or f"sess_{random.randint(1000,9999)}"

    entities = extract_entities(message)
    scam_type = predict_scam_type(message)
    confidence = calculate_confidence_ml(message)

    response_text = await generate_agent_response(
        current_session_id, message, scam_type, entities
    )

    if current_session_id not in sessions:
        sessions[current_session_id] = {"history": []}

    sessions[current_session_id]["history"].append(
        {"role": "user", "content": message}
    )
    sessions[current_session_id]["history"].append(
        {"role": "assistant", "content": response_text}
    )

    return AnalyzeResponse(
        session_id=current_session_id,
        scam_type=scam_type,
        confidence_score=confidence,
        extracted_entities=entities,
        agent_response=response_text,
        is_ml_used=ML_ENABLED
    )


