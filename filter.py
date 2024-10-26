import re
import sys
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from compute_accuracy import check, last_boxed_only_string, remove_boxed
from text_utils import format_cot, split_solutions


def main():
    filenames = sys.argv[2:]
    prior = []
    posterior = []
    for f in filenames:
        if "star-" in f:
            posterior.append(f)
        else:
            prior.append(f)
    print(prior)
    print(posterior)
    test_examples = load_dataset('json', data_files=prior, split="train")
    test_examples_pos = load_dataset('json', data_files=posterior, split="train")
    outs = []
    count = []
    limit = 32
    for example, example_pos in tqdm(zip(test_examples, test_examples_pos), total=len(test_examples)):
        count.append(0)
        tot = 0
        for one in example['output']:
            if tot >= limit: break
            equiv = check(one, example['solution'])
            if equiv:
                print("ok:", len(one.split()))
                tot += 1
                count[-1] = 1
                outs.append({"text": [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  example['problem']}, {"role": "assistant", "content": format_cot(one)}]})
        best = np.argmax(example['logprob'])
        best_e = example['output'][best]
        best = example['logprob'][best]
        if tot == 0:
            for one, p in zip(example_pos['star'], example_pos['logprob']):
                if tot >= limit: break
                equiv = check(one, example['solution'])
                if equiv:
                    print(np.exp(p)/np.exp(best))
                    print(len(best_e.split()))
                    #print(best_e)
                    #print('-'*50)
                    #print(one)
                    #print("="*50)
                    tot += 1
                    count[-1] = 1
                    outs.append({"text": [{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" +  example['problem']}, {"role": "assistant", "content": format_cot(one)}]})

    print("correct chains per example:", len(outs)/len(test_examples))
    print("examples that were correct:", sum(count))
    test_examples = Dataset.from_list(outs)
    test_examples.to_json(f"data/math/star_mixed_iteration{sys.argv[1]}_max{limit}.json")

if __name__ == '__main__':
    main()
