from transformers import pipeline

generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
output = generator("Share history of America", max_length=100)
print(output[0]['generated_text'])
