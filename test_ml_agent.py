import httpx
import sys

API_URL = "http://localhost:8000/analyze"
API_KEY = "hackathon-secret-key"

def test_ml_integration():
    print("Testing ML Integration...")
    
    # 1. Bank Scam
    try:
        resp = httpx.post(API_URL, headers={"x-api-key": API_KEY}, json={
            "message": "HDFC Alert: specific transaction of Rs 5000 debited. If not you, click http://bad.com"
        })
        data = resp.json()
        print(f"[Bank] Type: {data.get('scam_type')}, Conf: {data.get('confidence_score')}, ML Used: {data.get('is_ml_used')}")
        
        if data.get("scam_type") != "bank":
            print("❌ Bank Scam detection failed.")
            sys.exit(1)
            
        if not data.get("is_ml_used"):
            print("❌ ML Model was not used.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        sys.exit(1)

    # 2. Crypto Scam
    try:
        resp = httpx.post(API_URL, headers={"x-api-key": API_KEY}, json={
            "message": "Invest in Bitcoin and get double returns in 24 hours."
        })
        data = resp.json()
        print(f"[Crypto] Type: {data.get('scam_type')}, Conf: {data.get('confidence_score')}")
        
        if data.get("scam_type") != "investment":
             print("❌ Investment Scam detection failed.")
             sys.exit(1)

    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        sys.exit(1)
        
    print("\n✅ ML Integration Tests Passed!")

if __name__ == "__main__":
    test_ml_integration()
