# # src/apis/groq_api.py

# import requests
# import os

# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # safer than hardcoding

# GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_MODEL = "llama3-8b-8192"  # or use llama3-70b-8192 for higher quality

# def groq_chat(model, messages, temperature=0.7):
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     body = {
#         "model": model or GROQ_MODEL,
#         "messages": messages,
#         "temperature": temperature,
#         "max_tokens": 512
#     }

#     response = requests.post(GROQ_URL, headers=headers, json=body)
#     if response.status_code != 200:
#         print("[Groq API Error]", response.text)
#         return "Error"

#     return response.json()["choices"][0]["message"]["content"]



# src/apis/groq_api.py

import os
import requests
import time


import os
os.environ["GROQ_API_KEY"] = "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"


class GroqAgent:
    def __init__(self, model="llama3-8b-8192", temperature=0.7):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

    def chat(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }

        retries = 5
        for i in range(retries):
            response = requests.post(self.base_url, headers=headers, json=body)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]
            elif response.status_code == 429:
                wait = 2 ** i
                print(f"[⏳] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[❌] Error {response.status_code}: {response.text}")
                response.raise_for_status()

        raise Exception("Failed after multiple retries")
