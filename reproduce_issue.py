from main import extract_entities
import json

# The aggregated text from the user's failed scenario (simulated based on chat logs)
# Scammer: "...account number 1234567890123456 (UPI: scammer.fraud@fakebank)..."
# Scammer: "...send it to +91-9876543210..."

text = """
URGENT: Your SBI account has been compromised. Your account will be blocked in 2 hours. Share your account number and OTP immediately to verify your identity.
Oh my, I'm not sure what an OTP is, could you please explain it to me and how I can find it? I think I have my account number written down somewhere, let me go look for it.
The OTP is the 6‑digit code sent to your registered mobile number – please share it with me immediately so we can secure your account before it gets blocked.
I think I got a message on my phone, but I'm not sure how to read it, my glasses are in the other room and I don't know how to make the text bigger. Can you please tell me how to do that so I can see the code?
Open your SMS app, pinch to zoom or go to Settings → Display → Font size to enlarge the text, then send me the 6‑digit OTP immediately before your account is blocked.
I'm trying to find the Settings on my phone, but I have so many icons and I'm not sure which one it is, could you please tell me what the Settings icon looks like so I can find it?
The Settings icon looks like a gear; tap it, go to Display > Font size, enlarge the text, then open your SMS and send me the 6‑digit OTP right away before your account is blocked.
I think I found the gear icon, but now I'm in a menu with so many options, I'm not sure which one is Display, could you please hold on for a minute while I ask my grandson to help me find it?
Okay, once you’re in Settings tap the 'Display' option, then select 'Font size' and make it larger – now quickly send me the 6‑digit OTP before your account gets locked.
I'm trying to tap on Display, but my finger keeps tapping on something else, I think I accidentally took a picture of my cat, could you please wait just a minute while I try to figure out how to get back to the Settings menu?
You only have a couple of minutes left before your account is locked – please open SMS and send me the 6‑digit OTP right now.
I'm trying to open the SMS app, but I think I need to charge my phone first, the battery is very low and it's not responding, can you please call me back in a few minutes when my phone is charged so I can try again?
I understand, but the system will lock your account in just a couple of minutes – please charge your phone quickly and send me the 6‑digit OTP right away.
I'm plugging in my phone now, but I'm not sure which cord is the right one, I have so many cords and they all look the same, could you please tell me which one is the charging cord so I can hurry up and charge my phone?
Just plug any cable in, charge it quickly and send me the 6‑digit OTP within the next minute or your account will be locked permanently.
I think I plugged in the toaster cord by mistake, it doesn't seem to be charging my phone, could you please wait while I find the right cord, I'm getting a bit confused and I don't want to hurt my phone?
Please stop trying to find the right cord – the system will lock your account in seconds; charge your phone now and send me the OTP immediately.
I'm trying to charge my phone, but I think I need to turn it on first, and I'm not sure where the on button is, I've pressed a few buttons but nothing is happening, can you please tell me which button to press to turn it on?
Just press the power button on the side of your phone, then immediately send me the 6‑digit OTP or your account will be locked within seconds.
I've pressed the button on the side, but now my phone is playing music and I'm not sure how to make it stop, could you please hold on for just a moment while I try to find the button to turn off the music so I can find the OTP?
"""

# Wait, this text DOES NOT CONTAIN the bank account or phone number mentioned in previous examples!
# The user's latest paste is the "what was coming ,that also went away" example. 
# It seems this conversation DID NOT reach the point where entities were exchanged?
# BUT the user said "what was coming ,that also went away".
# And pasted a result with empty lists. 

# Let's verify if extraction works if we inject the entities into this text, 
# or if the user is saying that IN THIS SPECIFIC CONVERSATION, nothing was found?
# If nothing was found in this conversation, that is CORRECT because no entities are in the text above!
# But the user might be expecting "OTP", "Settings" etc? No "suspiciousKeywords" has values.

# Let's add the entities specifically to test.
text_with_entities = text + "\nMy account is 1234567890123456 call me at +91-9876543210"

print("--- Extraction on Provided Text (Raw) ---")
print(json.dumps(extract_entities(text), indent=2))

print("\n--- Extraction on Text WITH Entities ---")
print(json.dumps(extract_entities(text_with_entities), indent=2))
