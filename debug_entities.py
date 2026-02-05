from main import extract_entities
import json

# Text simulating the user's scenario
text = """
Scammer: send it to +91-9876543210
Honeypot: is my account number 1234567890123456?
Scammer: yes account 1234567890123456. Send to +91-9876543210.
"""

print("Testing Extraction...")
entities = extract_entities(text)
print(json.dumps(entities, indent=2))

# Validation
phones = entities["phoneNumbers"]
banks = entities["bankAccounts"]

# 1. 1234567890123456 should be BANK only
if "1234567890123456" in banks and not any("1234567890123456" in p for p in phones):
    print("PASS: Bank Account identified correctly.")
else:
    print("FAIL: Bank Account issue.")

# 2. +91-9876543210 should be PHONE only (normalized in logic, but raw in output)
# Check if 9876543210 is NOT in banks
if "9876543210" not in banks:
    print("PASS: Phone number NOT misclassified as Bank Account.")
else:
    print("FAIL: Phone number misclassified as Bank Account.")

# 3. 7890123456 (substring of bank) should NOT be in phones
if "7890123456" not in phones:
    print("PASS: Substring of bank account NOT misclassified as Phone.")
else:
    print("FAIL: Substring issue.")
