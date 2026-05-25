import pandas as pd
import numpy as np
from collections import defaultdict
import random
import subprocess

subprocess.run(['wget', 'https://github.com/CGCL-codes/Amain/raw/refs/heads/main/dataset/GCJ/googlejam4.zip'])
subprocess.run(['wget', 'https://github.com/CGCL-codes/Amain/raw/refs/heads/main/dataset/GCJ/GCJ_clone.csv'])
subprocess.run(['mkdir', 'GCJ'])
subprocess.run(['unzip', 'googlejam4.zip', '-d', 'GCJ'])
np.random.seed(42)
random.seed(42)

# =============================================
# 1. Wczytanie i budowa grup
# =============================================
clones = pd.read_csv("GCJ/GCJ_clone.csv", header=None)
clones[0] = clones[0].astype(str)
clones[1] = clones[1].astype(str)

graph = defaultdict(set)
for _, row in clones.iterrows():
    x, y = row.iloc[0], row.iloc[1]
    graph[x].add(y)
    graph[y].add(x)

visited = set()
clone_index = {}
current_index = 0

def dfs(start, idx):
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node)
        clone_index[node] = idx
        for neigh in graph[node]:
            if neigh not in visited:
                stack.append(neigh)

for node in list(graph.keys()):
    if node not in visited:
        current_index += 1
        dfs(node, current_index)

codes_count = defaultdict(int)
for idx in clone_index.values():
    codes_count[idx] += 1

sorted_groups = sorted(codes_count.items(), key=lambda x: x[1])
train_indices = {sorted_groups[0][0], sorted_groups[1][0]}

test_indices = [idx for idx in codes_count if idx not in train_indices]

# =============================================
# 2. Lepsze zrównoważone wybieranie par
# =============================================
used_codes = set()
selected_clone_pairs = []
selected_non_clone_pairs = []

test_codes_by_group = defaultdict(list)
for code, idx in clone_index.items():
    if idx in test_indices:
        test_codes_by_group[idx].append(code)

# --- 1. Clone pairs - staramy się rozłożyć po grupach ---
all_clone_pairs = []
for _, row in clones.iterrows():
    x, y = row.iloc[0], row.iloc[1]
    if clone_index.get(x) in test_indices and clone_index.get(y) in test_indices:
        all_clone_pairs.append((x, y, clone_index[x], clone_index[y]))

random.shuffle(all_clone_pairs)

# Wybieramy 50 par klonów starając się nie dominować jednej grupy
group_clone_count = defaultdict(int)

for x, y, idx1, idx2 in all_clone_pairs:
    if x in used_codes or y in used_codes:
        continue
    # Ograniczamy ile razy grupa może być użyta w parach klonów
    if group_clone_count[idx1] < 12 and group_clone_count[idx2] < 12:   # limit na grupę
        selected_clone_pairs.append((x, y, idx1, idx2))
        used_codes.add(x)
        used_codes.add(y)
        group_clone_count[idx1] += 1
        group_clone_count[idx2] += 1
    if len(selected_clone_pairs) >= 50:
        break

print(f"Wybrano {len(selected_clone_pairs)} par klonów")

# --- 2. Non-clone pairs - balansujemy względem clone pairs ---
group_nonclone_count = defaultdict(int)

for _ in range(30000):
    idx1 = random.choice(test_indices)
    idx2 = random.choice(test_indices)
    if idx1 == idx2:
        continue

    code1 = random.choice(test_codes_by_group[idx1])
    code2 = random.choice(test_codes_by_group[idx2])

    if code1 in used_codes or code2 in used_codes:
        continue

    # Staramy się zrównoważyć liczebność indeksu
    if (group_nonclone_count[idx1] < group_clone_count[idx1] + 3 and 
        group_nonclone_count[idx2] < group_clone_count[idx2] + 3):
        
        selected_non_clone_pairs.append((code1, code2, idx1, idx2))
        used_codes.add(code1)
        used_codes.add(code2)
        group_nonclone_count[idx1] += 1
        group_nonclone_count[idx2] += 1

    if len(selected_non_clone_pairs) >= 50:
        break

print(f"Wybrano {len(selected_non_clone_pairs)} par non-klonów")

# =============================================
# 3. Zapis
# =============================================
clone_df = pd.DataFrame(selected_clone_pairs, columns=['code1', 'code2', 'code1_clone_index', 'code2_clone_index'])
clone_df['label'] = 1

non_df = pd.DataFrame(selected_non_clone_pairs, columns=['code1', 'code2', 'code1_clone_index', 'code2_clone_index'])
non_df['label'] = 0

test_dataset = pd.concat([clone_df, non_df], ignore_index=True)
test_dataset = test_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

test_dataset.to_csv("test_dataset_50_50.csv", index=False)

print("\n=== ZAPISANO ===")
print(f"Plik: test_dataset_50_50.csv")