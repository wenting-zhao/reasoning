import re
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
from compute_accuracy import check, first_boxed_only_string, remove_boxed


def main():
    test_examples = load_dataset('json', data_files=sys.argv[2:], split="train")
    outs = []
    count = []
    for example in tqdm(test_examples):
        count.append(0)
        for one in example['output']:
            one = one.split("Problem:")[0].strip()
            one = one.split("\n\n")[0].strip()
            equiv = check(one, example['solution'])
            if equiv:
                count[-1] = 1
                outs.append({"text": "Problem: " + example['problem'] + "\nSolution: " + one})
                #print(outs[-1]["text"])
                #print("-"*100)

    print("correct chains per example:", len(outs)/len(test_examples))
    print("examples that were correct:", sum(count))
    test_examples = Dataset.from_list(outs)
    print(f"data/math/star_correct_chain_iteration{sys.argv[1]}.json")
    #test_examples.to_json(f"data/math/star_correct_chain_iteration{sys.argv[1]}.json")

if __name__ == '__main__':
    main()
