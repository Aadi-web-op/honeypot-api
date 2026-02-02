from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_full_flow():
    api_key = "hackathon-secret-key"
    
    # Test 1: Bank Scam (High Confidence)
    payload_bank = {
        "message": "URGENT: Your SBI account is blocked. Update PAN immediately via http://bit.ly/fake to avoid suspension.",
        "session_id": "test_bank_1"
    }
    res = client.post("/analyze", json=payload_bank, headers={"x-api-key": api_key})
    assert res.status_code == 200
    data = res.json()
    
    print("\n--- Bank Scam Test ---")
    print(f"Type: {data['scam_type']}")
    print(f"Confidence: {data['confidence_score']}")
    print(f"Response: {data['agent_response']}")
    print(f"Entities: {data['extracted_entities']}")
    
    assert data["scam_type"] == "bank"
    assert data["confidence_score"] > 0.5  # Expect high confidence (URL + keywords + urgent)
    assert "blocked" in data["agent_response"].lower() or "help" in data["agent_response"].lower() or "weird" in data["agent_response"].lower()
    
    # Test 2: Tech Support (Specific Type)
    payload_tech = {
        "message": "Microsoft Alert: Virus detected. Call +91 9876543210 to fix. install AnyDesk.",
    }
    res = client.post("/analyze", json=payload_tech, headers={"x-api-key": api_key})
    data = res.json()
    
    print("\n--- Tech Support Test ---")
    print(f"Type: {data['scam_type']}")
    print(f"Response: {data['agent_response']}")
    
    assert data["scam_type"] == "tech_support"
    assert "anydesk" in data["agent_response"].lower() or "computer" in data["agent_response"].lower() or "virus" in data["agent_response"].lower()

    # Test 3: Session Persistence (Check ID return)
    assert data["session_id"] is not None
    
    # Test 4: Auth Failure
    res = client.post("/analyze", json=payload_bank, headers={"x-api-key": "wrong"})
    assert res.status_code == 401
    print("\n--- Auth Test Passed (401) ---")

if __name__ == "__main__":
    test_full_flow()
