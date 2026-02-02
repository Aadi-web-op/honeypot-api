import asyncio
import httpx
import time

API_URL = "http://localhost:8000/analyze"
API_KEY = "hackathon-secret-key"

async def send_request(client, name, payload, api_key=API_KEY):
    headers = {"x-api-key": api_key} if api_key else {}
    start = time.time()
    try:
        response = await client.post(API_URL, json=payload, headers=headers)
        duration = time.time() - start
        return name, response.status_code, response.json() if response.status_code != 401 else response.text, duration
    except Exception as e:
        return name, 0, str(e), 0

async def run_evaluator_test():
    async with httpx.AsyncClient() as client:
        tasks = []
        
        # 1. Auth Test (Should Fail)
        tasks.append(send_request(client, "Auth Fail", {"message": "hi"}, api_key="wrong"))
        
        # 2. Empty Message (Validation Error)
        tasks.append(send_request(client, "Validation Fail", {})) # Missing 'message'
        
        # 3. Mixed Scam (Bank + Urgency)
        tasks.append(send_request(client, "Bank Scam", {
            "message": "ALERT: Your account ending in 8822 is blocked. Click http://bad.com immediately.",
            "session_id": "eval_1"
        }))
        
        # 4. Concurrent Load (5 requests)
        for i in range(5):
            tasks.append(send_request(client, f"Load Test {i}", {
                "message": f"I won a lottery {i}? call +91987654321{i}",
                "session_id": f"load_{i}"
            }))

        print("--- Starting Evaluator Stress Test ---")
        results = await asyncio.gather(*tasks)
        
        for name, status, data, duration in results:
            print(f"[{name}] Status: {status} | Time: {duration:.3f}s")
            if status == 200:
                print(f"   -> Type: {data.get('scam_type')} | Conf: {data.get('confidence_score')}")
            elif status != 200:
                print(f"   -> Response: {data}")

if __name__ == "__main__":
    asyncio.run(run_evaluator_test())
