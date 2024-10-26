from datasets import load_dataset
from compute_accuracy import remove_boxed, normalize_final_answer, last_boxed_only_string, is_equiv
from statistics import mode
from collections import defaultdict
import sys

ds = load_dataset("json", data_files=sys.argv[1], split="train")
solutions = list(load_dataset("hendrycks/competition_math", split="test")['solution'])
types = list(load_dataset("hendrycks/competition_math", split="test")['type'])
levels = list(load_dataset("hendrycks/competition_math", split="test")['level'])
correct = 0
cat2acc = defaultdict(list)
level2acc = defaultdict(list)
for i, one in enumerate(ds):
    if isinstance(one['output'], list):
        majority = []
        for x in one['output']:
            x = remove_boxed(last_boxed_only_string(x))
            if x is not None:
                x = normalize_final_answer(x)
            majority.append(x)
        output = mode(majority)
    else:
        output = remove_boxed(last_boxed_only_string(one['output']))
    answer = remove_boxed(last_boxed_only_string(solutions[i]))
    if answer is not None:
        answer = normalize_final_answer(answer)
    if output is not None:
        output = normalize_final_answer(output)
    equiv = is_equiv(output, answer)
    if equiv:
        correct += 1
        cat2acc[types[i]].append(1)
        level2acc[levels[i]].append(1)
    else:
        cat2acc[types[i]].append(0)
        level2acc[levels[i]].append(0)
print("math accuracy:", correct / len(ds))
for key in cat2acc:
    print(key, sum(cat2acc[key])/len(cat2acc[key]), len(cat2acc[key]))
print()
for key in sorted(level2acc):
    print(key, sum(level2acc[key])/len(level2acc[key]))

