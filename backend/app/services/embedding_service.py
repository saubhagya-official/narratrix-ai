from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

print("🌐 Using Hugging Face Inference API")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = []

    for text in texts:
        result = client.feature_extraction(text, model=MODEL_NAME)

        embeddings.append(result)

    return embeddings
