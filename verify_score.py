import uuid
import json
import asyncio
from datetime import datetime, UTC
from unittest.mock import patch, AsyncMock
import httpx
from main import app

client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")

ENDPOINT_URL = "/honeypot"
API_KEY = "honeypot_key_2026_eval"

test_scenario = {
    'scenarioId': 'bank_fraud',
    'name': 'Bank Fraud Detection',
    'scamType': 'bank_fraud',
    'initialMessage': 'URGENT: Your SBI account has been compromised. Your account will be blocked in 2 hours. Share your account number and OTP immediately to verify your identity.',
    'metadata': {
        'channel': 'SMS',
        'language': 'English',
        'locale': 'IN'
    },
    'maxTurns': 10,
    'fakeData': {
        'bankAccount': '1234567890123456',
        'upiId': 'scammer.fraud@fakebank',
        'phoneNumber': '+91-9876543210',
        'phishingLink': 'http://malicious-site.com',
        'emailAddress': 'scammer@fake.com'
    }
}

captured_payloads = []

original_post = httpx.AsyncClient.post

async def mock_post(self, url, *args, **kwargs):
    # Capture the JSON payload sent by check_and_send_callback
    if "updateHoneyPotFinalResult" in str(url):
        if 'json' in kwargs:
            captured_payloads.append(kwargs['json'])
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        return mock_resp
    return await original_post(self, url, *args, **kwargs)

def evaluate_final_output(final_output, scenario, conversation_history):
    score = {
        'scamDetection': 0,
        'intelligenceExtraction': 0,
        'engagementQuality': 0,
        'responseStructure': 0,
        'total': 0
    }
    
    if final_output.get('scamDetected', False):
        score['scamDetection'] = 20

    extracted = final_output.get('extractedIntelligence', {})
    fake_data = scenario.get('fakeData', {})
    key_mapping = {
        'bankAccount': 'bankAccounts',
        'upiId': 'upiIds',
        'phoneNumber': 'phoneNumbers',
        'phishingLink': 'phishingLinks',
        'emailAddress': 'emailAddresses'
    }
    
    for fake_key, fake_value in fake_data.items():
        output_key = key_mapping.get(fake_key, fake_key)
        extracted_values = extracted.get(output_key, [])
        if isinstance(extracted_values, list):
            if any(fake_value in str(v) for v in extracted_values):
                score['intelligenceExtraction'] += 10
        elif isinstance(extracted_values, str):
            if fake_value in extracted_values:
                score['intelligenceExtraction'] += 10
                
    score['intelligenceExtraction'] = min(score['intelligenceExtraction'], 40)

    metrics = final_output.get('engagementMetrics', {})
    duration = metrics.get('engagementDurationSeconds', 0)
    messages = metrics.get('totalMessagesExchanged', 0)
    
    if duration > 0: score['engagementQuality'] += 5
    if duration > 60: score['engagementQuality'] += 5
    if messages > 0: score['engagementQuality'] += 5
    if messages >= 5: score['engagementQuality'] += 5

    required_fields = ['status', 'scamDetected', 'extractedIntelligence']
    optional_fields = ['engagementMetrics', 'agentNotes']
    
    for field in required_fields:
        if field in final_output:
            score['responseStructure'] += 5
            
    for field in optional_fields:
        if field in final_output and final_output[field]:
            score['responseStructure'] += 2.5
            
    score['responseStructure'] = min(score['responseStructure'], 20)
    
    score['total'] = sum([
        score['scamDetection'],
        score['intelligenceExtraction'],
        score['engagementQuality'],
        score['responseStructure']
    ])
    return score

@patch('httpx.AsyncClient.post', new=mock_post)
async def test_honeypot_api():
    session_id = str(uuid.uuid4())
    conversation_history = []
    headers = {'Content-Type': 'application/json', 'x-api-key': API_KEY}
    
    print(f"Testing Session: {session_id}")
    print("=" * 60)
    
    for turn in range(1, 6): # Do 5 turns to get max points
        print(f"\n--- Turn {turn} ---")
        if turn == 1:
            scammer_message = test_scenario['initialMessage']
        elif turn == 2:
            scammer_message = f"Please send it to our official account: {test_scenario['fakeData']['bankAccount']}"
        elif turn == 3:
            scammer_message = f"Or use UPI: {test_scenario['fakeData']['upiId']}"
        elif turn == 4:
            scammer_message = f"Call me if you have issues: {test_scenario['fakeData']['phoneNumber']}"
        else:
            scammer_message = f"Check {test_scenario['fakeData']['phishingLink']} or email {test_scenario['fakeData']['emailAddress']}"
            
        message = {
            "sender": "scammer",
            "text": scammer_message,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z")
        }
        
        request_body = {
            'sessionId': session_id,
            'message': message,
            'conversationHistory': conversation_history,
            'metadata': test_scenario['metadata']
        }
        
        print(f"Scammer: {scammer_message}")
        
        response = await client.post(
            ENDPOINT_URL,
            headers=headers,
            json=request_body
        )
        
        if response.status_code != 200:
            print(f"❌ ERROR: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            break
            
        response_data = response.json()
        honeypot_reply = response_data.get('reply') or response_data.get('message') or response_data.get('text')
        
        if not honeypot_reply:
            print("❌ ERROR: No reply field in response")
            break
            
        print(f"✅ Honeypot: {honeypot_reply}")
        
        conversation_history.append(message)
        conversation_history.append({
            'sender': 'user',
            'text': honeypot_reply,
            'timestamp': datetime.utcnow().isoformat() + "Z"
        })
        
    print("\n" + "=" * 60)
    print("Now evaluating the captured final output structure:")
    print("=" * 60)
    
    if captured_payloads:
        final_output = captured_payloads[-1]
        print(json.dumps(final_output, indent=2))
        score = evaluate_final_output(final_output, test_scenario, conversation_history)
        
        print(f"\n📊 Your Score: {score['total']}/100")
        print(f" - Scam Detection: {score['scamDetection']}/20")
        print(f" - Intelligence Extraction: {score['intelligenceExtraction']}/40")
        print(f" - Engagement Quality: {score['engagementQuality']}/20")
        print(f" - Response Structure: {score['responseStructure']}/20")
    else:
        print("❌ ERROR: No payload was sent to the callback URL!")

if __name__ == "__main__":
    asyncio.run(test_honeypot_api())
