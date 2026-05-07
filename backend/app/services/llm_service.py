from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

print("🌐 Using Hugging Face Inference API for LLM")


def generate_answer(context: str, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer questions using ONLY the provided context. "
                "Be concise. "
                "If the answer is not in the context, say: Not found in document."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=200, temperature=0.3
    )

    return completion.choices[0].message.content.strip()
