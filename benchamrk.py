import os
import time
import pickle
import requests
from openai import OpenAI

RESULTS_FILE = "results/results.pickle"

MODELS = [
    {
        "name": "CodeBERT",
        "hf": "microsoft/codebert-base"
    },
    {
        "name": "gemma-4-31B-it",
        "hf": "google/gemma-4-31b-it"
    },
    {
        "name": "gemma-4-E4B-it",
        "hf": "google/gemma-4-8b-it"
    },
    {
        "name": "Qwen3.5-122B-A10B",
        "hf": "Qwen/Qwen2.5-72B-Instruct"
    },
    {
        "name": "Qwen3-Coder-Next",
        "hf": "Qwen/Qwen2.5-Coder-32B-Instruct"
    },
    {
        "name": "Mistral-Medium-3.5-128B",
        "hf": "mistralai/Mixtral-8x22B-Instruct-v0.1"
    },
    {
        "name": "INTELLECT-3",
        "hf": "PrimeIntellect/INTELLECT-1-Instruct"
    },
    {
        "name": "GPT-OSS-120B",
        "hf": "openai/gpt-oss-120b"
    },
    {
        "name": "GPT-OSS-20B",
        "hf": "openai/gpt-oss-20b"
    }
]

with open("dataset_creation/prompts.pickle", "rb") as f:
    prompts = pickle.load(f)

results = []

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

for model in MODELS:

    model_name = model["name"]

    print(f"\n=== MODEL {model_name} ===")

    for idx, prompt in enumerate(prompts):

        try:
            start = time.time()

            response = client.chat.completions.create(
                model=model["hf"],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2048
            )

            latency = time.time() - start

            answer = response.choices[0].message.content

            item = {
                "model": model_name,
                "hf_model": model["hf"],
                "prompt_id": idx,
                "prompt": prompt,
                "response": answer,
                "latency": latency,
                "timestamp": time.time()
            }

            results.append(item)

            with open(RESULTS_FILE, "wb") as f:
                pickle.dump(results, f)

            print(f"[OK] prompt={idx} latency={latency:.2f}s")

        except Exception as e:

            item = {
                "model": model_name,
                "prompt_id": idx,
                "error": str(e),
                "timestamp": time.time()
            }

            results.append(item)

            with open(RESULTS_FILE, "wb") as f:
                pickle.dump(results, f)

            print(f"[ERROR] {e}")