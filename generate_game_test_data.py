from datasets import load_dataset, Dataset
from tqdm import tqdm
from sample-model-b import gen_prompt, fewshot_examples


def main():
    test_examples = load_dataset("hendrycks/competition_math", split="test")
    outs = []
    for example in tqdm(test_examples):
        example["output"] = ["Sub-problem: aaaaaaa\nSolution to the sub-problem: qqqq"]
        prompt = gen_prompt(example, fewshot_examples)
        cut = prompt[0].find(example["problem"])
        outs.append({"text": prompt[0][cut:cut+len(example["problem"])] + "\nSub-problem:"})
    
    test_examples = Dataset.from_list(outs)
    test_examples.to_json(f"data/math/competition_math-game-test.json")

if __name__ == '__main__':
    main()
