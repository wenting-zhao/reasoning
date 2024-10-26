import re
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
from compute_accuracy import check, last_boxed_only_string, remove_boxed
from text_utils import format_cot, split_solutions
from collections import defaultdict


def main():
    test_examples = load_dataset('json', data_files=sys.argv[2:], split="train")
    types = list(load_dataset("hendrycks/competition_math", split="train")['type'])
    outs = []
    count = []
    cat2examples = defaultdict(list)
    limit = 32
    for i, example in tqdm(enumerate(test_examples)):
        count.append(0)
        key = 'output' if 'output' in example else 'star'
        tot = 0
        for one in example[key]:
            if tot >= limit: break
            preds = split_solutions(one)
            for pred in preds:
                equiv = check(pred, example['solution'])
                if equiv and len(pred.split()) < 500:
                    tot += 1
                    count[-1] = 1
                    outs.append({"text": [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  example['problem']}, {"role": "assistant", "content": format_cot(pred)}]})
                    cat2examples[types[i]].append({"text": [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  example['problem']}, {"role": "assistant", "content": format_cot(pred)}]})

    print("correct chains per example:", len(outs)/len(test_examples))
    print("examples that were correct:", sum(count))
    test_examples = Dataset.from_list(outs)
    test_examples.to_json(f"data/math/star_iteration{sys.argv[1]}_max{limit}.json")
    for key in cat2examples:
        test_examples = Dataset.from_list(cat2examples[key])
        test_examples.to_json(f"data/math/star_iteration{sys.argv[1]}_max{limit}_{key}.json".replace(" ", "_"))

if __name__ == '__main__':
    main()
