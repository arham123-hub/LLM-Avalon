from ...agents.hf_agent import HuggingFaceAgent

class HFExtractor:
    def __init__(self, extractor_name: str, model_name: str, extract_prompt: str,
                 system_prompt: str = "", temperature: float = 0.0,
                 few_shot_demos: list = None, output_dir: str = None):
        self.extractor_name = extractor_name
        self.model_name = model_name  # kept for consistency
        self.extract_prompt = extract_prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.few_shot_demos = few_shot_demos or []
        self.output_dir = output_dir
        self.llm = HuggingFaceAgent()

    def extract(self, input_text: str):
        # Format prompt with few-shot if needed
        prompt = ""
        if self.system_prompt:
            prompt += f"System: {self.system_prompt}\n"
        for demo in self.few_shot_demos:
            prompt += f"User: {demo['input']}\nAssistant: {demo['output']}\n"
        prompt += f"User: {self.extract_prompt.format(input_text)}\nAssistant:"

        response = self.llm.generator(prompt, max_new_tokens=64, temperature=self.temperature, do_sample=False)
        return response[0]["generated_text"].split("Assistant:")[-1].strip()
    
    @classmethod
    def init_instance(cls, **kwargs):
        return cls(**kwargs)
