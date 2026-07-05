import os
from dotenv import load_dotenv
from groq import Groq

# Load variables from .env
load_dotenv()

# Read API key
api_key = os.getenv("GROQ_API_KEY")

# Check if key exists
if not api_key:
    print("❌ API key not found!")
    exit()

print("✅ API key loaded successfully!")

# Create Groq client
client = Groq(api_key=api_key)

# Send a test prompt
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Give one sentence of advice for wheat irrigation."
        }
    ]
)

print("\n🌾 Groq Response:")
print(response.choices[0].message.content)