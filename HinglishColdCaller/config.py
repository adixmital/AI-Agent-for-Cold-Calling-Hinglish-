import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Use a conversational AI model
HF_MODEL = "meta-llama/Meta-Llama-3-8B"  # Alternative: "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
