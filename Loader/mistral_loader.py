from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import torch
import os


class MistralLoader:
    def __init__(self, model_name: str, use_auth_token: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=use_auth_token
        )
        self.model = AutoModel.from_pretrained(
            model_name, use_auth_token=use_auth_token
        )

    def get_embeddings(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state


if __name__ == "__main__":
    model = AutoModel.from_pretrained("Bharat05092003/mistral_llm_v3")
    use_auth_token = os.getenv(
        "HUGGINGFACE_TOKEN"
    )  # Ensure you have set this environment variable
    loader = MistralLoader(model, use_auth_token)
    text = "This is a sample text for embedding."
    embeddings = loader.get_embeddings(text)
    print(embeddings)
