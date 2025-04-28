# from transformers import pipeline

# class HuggingFaceAgent:
#     def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
#         # Load locally installed model
#         self.generator = pipeline("text-generation", model=model_name, device=-1)

#     def chat(self, messages, max_tokens=512, temperature=0.7):
#         """
#         Simulates the OpenAI ChatCompletion API call using HuggingFace model.
#         Takes in a list of messages like:
#         [
#             {"role": "system", "content": "..."},
#             {"role": "user", "content": "..."},
#             {"role": "assistant", "content": "..."},
#         ]
#         """
#         prompt = ""
#         for msg in messages:
#             role = msg['role'].capitalize()
#             content = msg['content']
#             prompt += f"{role}: {content}\n"
#         prompt += "Assistant:"

#         # Generate output
#         # output = self.generator(prompt, max_length=max_tokens, temperature=temperature, truncation=True)
#         output = self.generator(prompt, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, truncation=True)


#         # Postprocess
#         response_text = output[0]['generated_text'].split("Assistant:")[-1].strip()
#         return {"role": "assistant", "content": response_text}




from transformers import pipeline
import torch

class HuggingFaceAgent:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=512):
        self.device = "cpu"
        print(f"[⚙️ Device set to use {self.device}]")

        # Set the correct device for pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if self.device == "mps" else -1
        )
        self.max_new_tokens = max_new_tokens

    def chat(self, messages, max_tokens=None, temperature=0.7):
        """
        Simulates the OpenAI ChatCompletion API call using HuggingFace model.
        Takes in a list of messages like:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
        Returns a dict: {"role": "assistant", "content": generated_text}
        """
        prompt = ""
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            prompt += f"{role}: {content}\n"
        prompt += "Assistant:"

        # Choose max tokens to generate
        max_new = max_tokens or self.max_new_tokens

        try:
            print(f"[🧠 Generating | Temp: {temperature} | Tokens: {max_new} | Device: {self.device}]")
            output = self.generator(
                prompt,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=True,
                truncation=True
            )
            response_text = output[0]['generated_text'].split("Assistant:")[-1].strip()
        except Exception as e:
            response_text = f"[Error generating response: {str(e)}]"

        return {"role": "assistant", "content": response_text}
