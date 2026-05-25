import os
import csv
import pickle

# ============================================
# CONFIG
# ============================================

PROMPTS_DIR = "prompts"
CSV_FILE = "test_dataset_50_50.csv"
CODE_DIR = "GCJ/googlejam4"

OUTPUT_FILE = "prompts.pickle"

# ============================================
# LOAD PROMPT TEMPLATES (CODE_1 / CODE_2)
# ============================================

pair_templates = []

description_template = None

for filename in sorted(os.listdir(PROMPTS_DIR)):

    filepath = os.path.join(PROMPTS_DIR, filename)

    if not os.path.isfile(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        template = f.read()

    name_lower = filename.lower()

    if "description" in name_lower:
        description_template = {
            "template_name": filename,
            "template": template
        }
    else:
        pair_templates.append({
            "template_name": filename,
            "template": template
        })

print(f"Loaded {len(pair_templates)} pair templates")
print(f"Description template: {description_template is not None}")

# ============================================
# LOAD CSV
# ============================================

pairs = []

with open(CSV_FILE, "r", encoding="utf-8") as f:

    reader = csv.reader(f)
    # skip headers
    next(reader, None)

    for row_id, row in enumerate(reader):

        if len(row) < 5:
            continue

        code1_name = row[0].strip()
        code2_name = row[1].strip()
        is_clone = row[4].strip()

        if is_clone.lower() in ["1", "true", "yes"]:
            is_clone = True
        elif is_clone.lower() in ["0", "false", "no"]:
            is_clone = False

        pairs.append({
            "code1_name": code1_name,
            "code2_name": code2_name,
            "is_clone": is_clone
        })

print(f"Loaded {len(pairs)} pairs")

# ============================================
# LOAD CODE
# ============================================

def load_code(filename):
    path = os.path.join(CODE_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ============================================
# BUILD PROMPTS
# ============================================

all_prompts = []

for pair_id, pair in enumerate(pairs):

    code1_name = pair["code1_name"]
    code2_name = pair["code2_name"]
    is_clone = pair["is_clone"]

    code1 = load_code(code1_name)
    code2 = load_code(code2_name)

    # --------------------------------------------
    # 1. TEMPLATES: CODE_1 / CODE_2
    # --------------------------------------------

    for tpl in pair_templates:

        final_prompt = (
            tpl["template"]
            .replace("{CODE_1}", code1)
            .replace("{CODE_2}", code2)
        )

        all_prompts.append({
            "pair_id": pair_id,
            "type": "pair_template",
            "template_name": tpl["template_name"],
            "code1_name": code1_name,
            "code2_name": code2_name,
            "is_clone": is_clone,
            "prompt": final_prompt
        })

    # --------------------------------------------
    # 2. DESCRIPTION: osobno dla CODE1 i CODE2
    # --------------------------------------------

    if description_template is not None:

        # CODE1
        prompt1 = description_template["template"].replace("{CODE}", code1)

        all_prompts.append({
            "pair_id": pair_id,
            "type": "description",
            "which_code": "code1",
            "template_name": description_template["template_name"],
            "code_name": code1_name,
            "is_clone": is_clone,
            "prompt": prompt1
        })

        # CODE2
        prompt2 = description_template["template"].replace("{CODE}", code2)

        all_prompts.append({
            "pair_id": pair_id,
            "type": "description",
            "which_code": "code2",
            "template_name": description_template["template_name"],
            "code_name": code2_name,
            "is_clone": is_clone,
            "prompt": prompt2
        })

# ============================================
# SAVE
# ============================================

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(all_prompts, f)

print(f"Saved {len(all_prompts)} prompts -> {OUTPUT_FILE}")