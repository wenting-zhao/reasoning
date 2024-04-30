import sys
import re

from datasets import load_dataset, Dataset
from compute_accuracy import remove_boxed, last_boxed_only_string

name = sys.argv[1]
split = sys.argv[2]
option = sys.argv[3]
ds = load_dataset(name, split=split)
outs = []
for one in ds:
    curr = [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  one['problem']}]
    if option == "answer":
        answer = remove_boxed(last_boxed_only_string(one['solution']))
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
