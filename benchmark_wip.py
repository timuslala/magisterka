import os
import time
import pickle



with open("dataset_creation/prompts.pickle", "rb") as f:
    prompts = pickle.load(f)


for idx, prompt in enumerate(prompts):
    print(prompt)
    exit(0)