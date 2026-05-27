import os
import sys
import time
import pickle
from openai import OpenAI

RESULTS_DIR = "results"

MODELS = [
    {
        "name": "gemma-4-31B-it",
        "hf": "google/gemma-4-31b-it"
    },
    {
        "name": "gemma-4-E4B-it",
        "hf": "google/gemma-4-E4B-it"
    },
    {
        "name": "Qwen3.5-122B-A10B",
        "hf": "Qwen/Qwen3.5-122B-A10B"
    },
    {
        "name": "Qwen3-Coder-Next",
        "hf": "Qwen/Qwen3-Coder-Next"
    },
    {
        "name": "Mistral-Medium-3.5-128B",
        "hf": "mistralai/Mistral-Medium-3.5-128B"
    },
    {
        "name": "INTELLECT-3",
        "hf": "PrimeIntellect/INTELLECT-3"
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

# -------------------------
# CLI ARG: model["hf"]
# -------------------------
if len(sys.argv) < 2:
    print("Usage: python script.py <model_hf>")
    sys.exit(1)

target_hf = sys.argv[1]

# find model
model = next((m for m in MODELS if m["hf"] == target_hf), None)

if model is None:
    print(f"Model not found: {target_hf}")
    print("Available models:")
    for m in MODELS:
        print(" -", m["hf"])
    sys.exit(1)

model_name = model["name"]
model_hf = model["hf"]

output_file = os.path.join(RESULTS_DIR, f"{model_name}reversed.pickle")

os.makedirs(RESULTS_DIR, exist_ok=True)

# load prompts
with open("dataset_creation/prompts.pickle", "rb") as f:
    prompts = pickle.load(f)

results = []

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

print(f"\n=== RUNNING MODEL: {model_name} ({model_hf}) ===")

for idx, prompt in reversed(list(enumerate(prompts))):

    try:
        start = time.time()

        response = client.chat.completions.create(
            model=model_hf,
            messages=[
                {
                    "role": "user",
                    "content": prompt["prompt"]
                }
            ],
            temperature=0.2
        )

        latency = time.time() - start
        answer = response.choices[0].message.content

        item = {
            "model": model_name,
            "hf_model": model_hf,
            "prompt_id": idx,
            "prompt": prompt["prompt"],
            "response": answer,
            "latency": latency,
            "timestamp": time.time(),
            "is_clone": prompt["is_clone"]
        }

        results.append(item)

        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        print(f"[OK] prompt={idx} latency={latency:.2f}s")

    except Exception as e:

        item = {
            "model": model_name,
            "hf_model": model_hf,
            "prompt_id": idx,
            "error": str(e),
            "timestamp": time.time()
        }

        results.append(item)

        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        print(f"[ERROR] {e}")