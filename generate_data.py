import sys
import re

from datasets import load_dataset, Dataset
from compute_accuracy import check, remove_boxed, last_boxed_only_string
from text_utils import format_qa

name = sys.argv[1:-2]
split = sys.argv[-2]
option = sys.argv[-1]
upper = 2
if name[0].endswith("json"):
    ds = load_dataset('json', data_files=name, split="train")
else:
    ds = load_dataset(name[0], split=split)
outs = []
for one in ds:
    if option == "answer":
        curr = [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  one['problem']}]
        answer = remove_boxed(last_boxed_only_string(one['solution']))
        if answer is None:
            continue
        curr.append({"role": "assistant", "content": '\\boxed{'+answer+'}'})
        curr = {"text": curr}
        outs.append(curr)
    elif option == "cot":
        curr = [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  one['problem']}]
        curr.append({"role": "assistant", "content": one['solution']})
        curr = {"text": curr}
        outs.append(curr)
    elif option == "game":
        if split == "train":
            count = 0
            for x in one['model-b']:
                equiv = check(x, one['solution'])
                if equiv:
                    curr = [{"role": "user", "content": "Solve the following math problem by decomposing it into a sequence of sub-problems and solving each at a time.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  one['problem']}]
                    out = format_qa(x)
                    if len(out) < 1:
                        print("WARNING:", x)
                        print('-'*100)
                    res = ""
                    for q, a in out:
                        res += f"Sub-problem: {q}\nSolution to the sub-problem: {a}\n"
                    res += "Therefore, the answer is \\boxed{"+remove_boxed(last_boxed_only_string(one['solution']))+"}."
                    curr.append({"role": "assistant", "content": res})
                    curr = {"text": curr}
                    outs.append(curr)
                    count += 1
                    if count >= upper:
                        break
        else:
            curr = [{"role": "user", "content": "Solve the following math problem by decomposing it into a sequence of sub-problems and solving each at a time.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  one['problem']}]
            curr.append({"role": "assistant", "content": one['solution']})
            curr = {"text": curr}
            outs.append(curr)
if name[0].endswith("json"):
    if "competition_math" in name[0]:
        out_name = f"data/math/competition_math-{split}-{option}.json"
else:
    out_name = f"data/math/{sys.argv[1].split('/')[-1]}-{split}-{option}.json"
if option == "game":
    out_name = out_name.replace('.json', f'_max{upper}.json')
out_ds = Dataset.from_list(outs)
out_ds.to_json(out_name)
