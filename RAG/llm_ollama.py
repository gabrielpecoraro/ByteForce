import requests

class OllamaLLM:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt, temperature=0.5, max_tokens=512):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "num_predict": max_tokens
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise Exception(f"Ollama error {response.status_code}: {response.text}")
