import re
import sys
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from compute_accuracy import check, last_boxed_only_string, remove_boxed
from text_utils import format_cot, split_solutions


def main():
    test_examples = load_dataset('json', data_files=sys.argv[1], split="train")
    outs = []
    for example in tqdm(test_examples):
        best = np.argmax(example['logprob'])
        best_e = example['output'][best]
        outs.append({"text": [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  example['problem']}, {"role": "assistant", "content": format_cot(best_e)}]})
    test_examples = Dataset.from_list(outs)
    test_examples.to_json(f"data/math/posterior_filtering.json")

if __name__ == '__main__':
    main()
