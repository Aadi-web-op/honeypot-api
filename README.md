# Honeypot API

A hackathon-ready API that analyzes scam messages, extracts intelligence, and provides agentic responses to waste scammers' time.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## API

### POST /analyze

Headers:
- `x-api-key`: `hackathon-secret-key`

Body:
```json
{
  "message": "Congratulations! You won a lottery. Send money to upi@bank."
}
```
