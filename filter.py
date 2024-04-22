import re
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
from check_equivalence import check, first_boxed_only_string, remove_boxed
from asymmetric_filtering import gen_prompt, fewshot_examples


def main():
    test_examples = load_dataset('json', data_files=sys.argv[1:], split="train")
    outs = []
    outs_model_a = []
    for example in tqdm(test_examples):
        for one, steps in zip(example['model-b'], example['steps']):
            one = one.split("Problem:")[0].strip()
            one = one.split("\n\n")[0].strip()
            equiv = check(one, example['solution'])
            if equiv:
                new = {"problem": example["problem"], "solution": example["solution"], "output": [one]}
                prompt = gen_prompt(new, fewshot_examples)
                cut = prompt[0].find(example["problem"])
                if len(prompt) > 0:
                    outs.append({"text": prompt[0][cut:] + one})
                    answer = remove_boxed(first_boxed_only_string(example["solution"]))
                    text = "Problem: " + example['problem'] + f"\nSolution: {answer}\n\n"
                    for q, a in steps:
                        text += f"Sub-problem: {q}\nSolution to the sub-problem: {a}\n"
                    text = text.strip()
                    text += f"\nTherefore, the answer is {answer}."
                    outs_model_a.append({"text": text})
                    print(outs_model_a[-1]["text"])
    
    print("correct chains per example:", len(outs)/len(test_examples))
    test_examples = Dataset.from_list(outs)
    test_examples.to_json("data/math/correct_chain_iteration0.json")
    test_examples = Dataset.from_list(outs_model_a)
    test_examples.to_json("data/math/model-a-correct_chain_iteration0.json")

if __name__ == '__main__':
    main()
