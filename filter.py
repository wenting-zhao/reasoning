import re
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
from check_equivalence import check
from asymmetric_filtering import gen_prompt, fewshot_examples


def main():
    test_examples = load_dataset('json', data_files=sys.argv[1:], split="train")
    outs = []
    for example in tqdm(test_examples):
        for one in example['model-b']:
            one = one.split("Problem:")[0].strip()
            one = one.split("\n\n")[0].strip()
            equiv = check(one, example['solution'])
            if equiv:
                new = {"problem": example["problem"], "solution": example["solution"], "output": [one]}
                prompt = gen_prompt(new, fewshot_examples)
                cut = prompt[0].find(example["problem"])
                if len(prompt) > 0:
                    outs.append({"text": prompt[0][cut:] + one})
    
    print("correct chains per example:", len(outs)/len(test_examples))
    test_examples = Dataset.from_list(outs)
    test_examples.to_json("data/math/correct_chain_iteration0.json")

if __name__ == '__main__':
    main()
