import os
from dotenv import load_dotenv
from groq import Groq

# 1. Try loading .env
print("Loading .env...")
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found in environment variables.")
else:
    print(f"SUCCESS: Found API Key: {api_key[:10]}...")

# 2. Try initializing Groq client
try:
    print("Initializing Groq client...")
    client = Groq(api_key=api_key)
    
    print("Sending test request to Groq...")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say 'Hello' if you can hear me.",
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    print("Response from Groq:")
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"ERROR: Failed to connect to Groq. Details: {e}")
