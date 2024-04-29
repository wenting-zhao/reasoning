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

name = sys.argv[1]
split = sys.argv[2]
option = sys.argv[3]
ds = load_dataset(name, split=split)
outs = []
for one in ds:
    curr = [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  one['problem']}]
    if option == "answer":
        answer = extract_substrings(one['solution'])
        if answer is None:
            continue
        curr.append({"role": "assistant", "content": '\\boxed{'+answer+'}'})
    elif option == "cot":
        curr.append({"role": "assistant", "content": one['solution']})
    curr = {"text": curr}
    outs.append(curr)
out_name = f"data/math/{sys.argv[1].split('/')[-1]}-{split}-{option}.json"
out_ds = Dataset.from_list(outs)
out_ds.to_json(out_name)
