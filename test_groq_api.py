from src.apis.groq_api import GroqAgent

agent = GroqAgent(model="llama3-8b-8192")
response = agent.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])
print(response["content"])
