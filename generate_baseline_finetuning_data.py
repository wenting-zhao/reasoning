import sys
import re

from datasets import load_dataset, Dataset

def extract_substrings(text):
    parts = text.split(r"\boxed")
    matches = []
    for part in parts[1:]:  # Skip the first part as it does not start with \boxed
        if part.startswith("{"):
            brace_level = 0
            for i, char in enumerate(part):
                if char == "{":
                    brace_level += 1
                elif char == "}":
                    brace_level -= 1
                    if brace_level == 0:
                        matches.append(
                            part[1:i]
                        )  # Extract the content inside the braces
                        break

    if len(matches) == 0:
        return None

    return matches[0]

split = sys.argv[3]
ds = load_dataset(sys.argv[1], split=split)
option = sys.argv[2]
outs = []
for one in ds:
    curr = one['problem']
    if split == "train":
        if option == "answer":
            answer = extract_substrings(one['solution'])
            if answer is None:
                continue
            curr += '\\boxed{'+answer+'}'
        elif option == "cot":
            curr += one['solution']
    outs.append({"text": curr})
out_name = f"data/math/{sys.argv[1].split('/')[-1]}-{option}.json"
if split=="test":
    out_name = out_name.replace('.json', '-test.json')
out_ds = Dataset.from_list(outs)
out_ds.to_json(out_name)
