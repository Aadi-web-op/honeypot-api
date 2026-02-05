import os
import re
import json
import logging
import random
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv
import joblib
from groq import Groq

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_KEY = "honeypot_key_2026_eval"
CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Data Models (Strictly matching the requirements) ---

class Message(BaseModel):
    sender: str
    text: str
    timestamp: int

class Metadata(BaseModel):
    channel: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None

class AnalyzeRequest(BaseModel):
    sessionId: str
    message: Message
    conversationHistory: List[Message] = []
    metadata: Optional[Metadata] = None

# --- Global Components ---

app = FastAPI()
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Global variables for models and clients
scam_classifier = None
tfidf_vectorizer = None
groq_client = None

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    global scam_classifier, tfidf_vectorizer, groq_client
    
    # 1. Load ML Models
    try:
        if os.path.exists("scam_classifier.pkl"):
            scam_classifier = joblib.load("scam_classifier.pkl")
            logger.info("Loaded scam_classifier.pkl")
        else:
            logger.warning("scam_classifier.pkl not found. Falling back to keyword mode.")
            
        if os.path.exists("tfidf_vectorizer.pkl"):
            tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
            logger.info("Loaded tfidf_vectorizer.pkl")
        else:
            logger.warning("tfidf_vectorizer.pkl not found.")
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")

    # 2. Initialize Groq Client
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized.")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")
    else:
        logger.error("GROQ_API_KEY not set in environment.")

# --- Security ---

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key

# --- Helper Functions ---

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extracts entities using regex."""
    upi_pattern = r'[\w\.-]+@[\w\.-]+'
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    phone_pattern = r'(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}'
    bank_account_pattern = r'\b\d{9,18}\b'
    
    # Simple keywords for suspicious terms
    suspicious_keywords_list = ["urgent", "verify", "block", "suspend", "kyc", "pan", "aadhar", "win", "lottery", "expired"]
    found_keywords = [word for word in suspicious_keywords_list if word in text.lower()]

    return {
        "bankAccounts": re.findall(bank_account_pattern, text),
        "upiIds": re.findall(upi_pattern, text),
        "phishingLinks": re.findall(url_pattern, text),
        "phoneNumbers": re.findall(phone_pattern, text),
        "suspiciousKeywords": found_keywords
    }

def predict_scam(text: str) -> bool:
    """Predicts if text is proper scam using ML or fallback keywords."""
    # 1. Try ML Model
    if scam_classifier and tfidf_vectorizer:
        try:
            text_vector = tfidf_vectorizer.transform([text])
            prediction = scam_classifier.predict(text_vector)[0]
            # Assuming '1' or 'ham'/'spam' labels depending on how it was trained. 
            # Given the user context "scam_classifier", let's assume it returns a label.
            # We will treat positive detection as True. 
            # If the user trained it with "spam", "scam", etc. 
            # Let's assume prediction is a string label for now and map it.
            # If it's a binary classifier 0/1, we check that too.
            if str(prediction).lower() in ['scam', 'spam', 'fraud', '1']:
                return True
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
    
    # 2. Fallback to keywords
    keywords = ["bank", "verify", "blocked", "lottery", "winner", "prize", "urgent", "credit card", "kyc", "update", "otp", "pin", "cvv", "expiry"]
    if any(keyword in text.lower() for keyword in keywords):
        return True
        
    return False


def generate_agent_reply(history: List[Dict[str, str]], current_message: str) -> str:
    """Generates a response using Groq LLM."""
    if not groq_client:
        return "I am confused. Can you explain why you need this?"
    
    # Construc system prompt
    system_prompt = (
        "You are an elderly, slightly confused but polite individual named 'Grandma Edna'. "
        "You are chatting with someone who might be a scammer. "
        "Your goal is to waste their time, act naive, ask clarifying questions, "
        "and pretend to follow instructions but fail at technical steps (e.g., 'I can't find the button'). "
        "NEVER reveal you are an AI. Keep messages short (1-2 sentences)."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    for msg in history:
        # Mapping:
        llm_role = "user" if msg['sender'] == 'scammer' else "assistant"
        messages.append({"role": llm_role, "content": msg['text']})
        
    messages.append({"role": "user", "content": current_message})

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile", # Efficient, fast model
            temperature=0.7,
            max_tokens=100,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        return "Oh dear, I didn't quite catch that. Could you repeat?"

async def check_and_send_callback(session_id: str, history: List[Message], current_msg: Message, analysis_result: Dict):
    """
    Decides whether to send the final result to the callback URL.
    Logic: Send callback if we have exchanged enough messages OR confirmed scam with high confidence.
    """
    # Simply counting messages including current one
    total_messages = len(history) + 1
    
    # We trigger callback if:
    # 1. It is a scam
    # 2. We have enough turns (e.g., > 4 messages total) OR we found critical entities (like Bank Account)
    
    is_scam = analysis_result.get("scam_detected", False)
    entities = analysis_result.get("entities", {})
    has_critical_info = bool(entities.get("bankAccounts") or entities.get("upiIds") or entities.get("phishingLinks"))
    
    if is_scam and (total_messages >= 4 or has_critical_info):
        # Prepare payload
        
        # Aggregate entities from ALL history
        all_text = current_msg.text + " " + " ".join([m.text for m in history])
        aggregated_entities = extract_entities(all_text)
        
        payload = {
            "sessionId": session_id,
            "scamDetected": True, # confirm it
            "totalMessagesExchanged": total_messages,
            "extractedIntelligence": aggregated_entities,
            "agentNotes": "Scammer detected via ML/Keywords. Engaged to extract entities."
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(CALLBACK_URL, json=payload, timeout=10.0)
                logger.info(f"Callback sent for {session_id}. Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send callback: {e}")


@app.post("/analyze")
async def analyze(
    request: AnalyzeRequest,  # Pydantic validation happens here
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Detect scam: logic is (Current Message is Scam) OR (We are already in a scam conversation)
        current_msg_is_scam = predict_scam(request.message.text)
        has_history = len(request.conversationHistory) > 0
        
        # If we have history, we assume we are already in the honey-pot flow
        is_scam = current_msg_is_scam or has_history
        
        # 2. Extract Entities
        entities = extract_entities(request.message.text)
        
        if is_scam:
            # Convert history to simple dict list for helper
            history_dicts = [m.dict() for m in request.conversationHistory]
            agent_reply = generate_agent_reply(history_dicts, request.message.text)
        else:
            # If not detected as scam, return a standard message or maybe the agent still replies?
            # "The Agent continues the conversation" implies it only engages if scam detected.
            # But the API format demands a reply.
            agent_reply = "I don't think I am interested. Thank you."

        # 4. Schedule Callback (Fire and forget)
        if is_scam:
            analysis_data = {
                "scam_detected": True,
                "entities": entities
            }
            background_tasks.add_task(
                check_and_send_callback,
                request.sessionId,
                request.conversationHistory,
                request.message,
                analysis_data
            )

        return {
            "status": "success",
            "reply": agent_reply
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "Honeycomb API Active", "version": "2.0"}
