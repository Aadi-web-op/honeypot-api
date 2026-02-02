import csv
import random

SCAM_TYPES = ["bank", "lottery", "loan", "investment", "tech_support"]

TEMPLATES = {
    "bank": [
        "Your SBI account {account} is blocked. Update KYC at {url}.",
        "HDFC Alert: specific transaction of Rs {amount} debited. If not you, click {url}",
        "Dear Customer, your PAN card is linked to account {account}. Verify immediately.",
        "RBI Warning: Your account will be suspended today. Call {phone} to prevent this.",
        "ICICI Bank: KYC pending. Your debit card is blocked. Reactivate here: {url}"
    ],
    "lottery": [
        "Congrats! You won {amount} in the KBC Lottery. Call {phone} to claim.",
        "Your mobile number has won a prize of {amount}. Send details to {email}.",
        "Coca-Cola Award Winner! You have won a car and {amount}. Click {url}.",
        "Samsung Lucky Draw: You are the 1st prize winner. Contact {phone} now.",
        "Mega Bumper Prize! Rs {amount} credited to your wallet? Verify at {url}."
    ],
    "loan": [
        "Pre-approved Personal Loan of {amount} available. Interest 2%. Apply: {url}",
        "Instant cash loan up to 5 Lakhs. No CIBIL check. Call {phone}.",
        "Get a loan in 5 mins. Aadhar Card only. 0% Interest for first month.",
        "Your business loan of {amount} is approved. Pay processing fee to disburse.",
        "Need money? Instant transfer to bank. Click {url} to apply now."
    ],
    "investment": [
        "Invest Rs 5000 and get Rs {amount} in 3 days. Crypto Mining.",
        "Double your money in 24 hours. Guaranteed returns. Join WhatsApp group: {url}",
        "Stock Market Tips: 100% profit daily. Join our VIP channel.",
        "Trading opportunity! Earn {amount} daily from home. Ask me how.",
        "Bitcoin investment scheme. High returns. Limited slots available."
    ],
    "tech_support": [
        "Microsoft Windows Alert: Virus detected. Call {phone} immediately.",
        "Your computer is hacked. Contact Microsoft Support at {phone}.",
        "System Error 0x8024. Data loss imminent. Install AnyDesk and call us.",
        "Apple Support: Your iCloud is compromised. Call {phone} to reset.",
        "Warning: Spyware found on your PC. Do not shut down. Call {phone}."
    ]
}

def generate_row(scam_type):
    template = random.choice(TEMPLATES[scam_type])
    
    # Fill dynamic slots
    text = template.format(
        account="XX" + str(random.randint(1000, 9999)),
        amount=str(random.randint(10, 99)) + " Lakhs",
        url="http://bit.ly/" + str(random.randint(1000, 9999)),
        phone="+91 " + str(random.randint(7000000000, 9999999999)),
        email="claim@scam.com"
    )
    return text, scam_type

def main():
    with open("scam_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        
        # Balance dataset: 100 per class = 500 samples
        for stype in SCAM_TYPES:
            for _ in range(100):
                writer.writerow(generate_row(stype))
                
    print("Generated 500 samples in scam_dataset.csv")

if __name__ == "__main__":
    main()
