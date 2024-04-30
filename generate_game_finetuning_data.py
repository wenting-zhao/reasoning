import re
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
from compute_accuracy import check, last_boxed_only_string, remove_boxed
from text_utils import format_qa, format_cot
from sample_model_a import format_example as format_example_a
from sample_model_b import format_example as format_example_b


def main():
    test_examples = load_dataset('json', data_files=sys.argv[2:], split="train")
    outs_model_a = []
    outs_model_b = []
    count = []
    for example in tqdm(test_examples):
        count.append(0)
        for one, steps in zip(example['model-b'], example['model-a']):
            equiv = check(one, example['solution'])
            if equiv:
                count[-1] = 1
                qa = format_qa(steps)
                curr_a = {"problem": example["problem"], "steps": qa, "answer": remove_boxed(last_boxed_only_string(example["solution"]))}
                model_a = format_example_a(curr_a, include_answer=True)
                outs_model_a.append({"text": model_a})
                curr_b = {"problem": example["problem"], "steps": qa, "answer": remove_boxed(last_boxed_only_string(example["solution"]))}
                model_b = format_example_b(curr_b, include_answer=True)
                outs_model_b.append({"text": model_b})
                print(model_a)
    
    print("correct chains per example:", len(outs_model_b)/len(test_examples))
    print("examples that were correct:", sum(count))
    test_examples = Dataset.from_list(outs_model_a)
    test_examples.to_json(f"data/math/model-a_iteration{sys.argv[1]}.json")
    test_examples = Dataset.from_list(outs_model_b)
    test_examples.to_json(f"data/math/model-b_iteration{sys.argv[1]}.json")

if __name__ == '__main__':
    main()
