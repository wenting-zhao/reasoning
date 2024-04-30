import re
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
from compute_accuracy import check, last_boxed_only_string, remove_boxed
from text_utils import format_cot


def main():
    test_examples = load_dataset('json', data_files=sys.argv[2:], split="train")
    outs = []
    count = []
    for example in tqdm(test_examples):
        count.append(0)
        for one in example['output']:
            equiv = check(one, example['solution'])
            if equiv:
                count[-1] = 1
                outs.append({"text": [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  example['problem']}, {"role": "assistant", "content": format_cot(one)}]})

    print("correct chains per example:", len(outs)/len(test_examples))
    print("examples that were correct:", sum(count))
    test_examples = Dataset.from_list(outs)
    test_examples.to_json(f"data/math/star_iteration{sys.argv[1]}.json")

if __name__ == '__main__':
    main()
