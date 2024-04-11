import argparse
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    args = parser.parse_args()

    test_examples = load_dataset('json', data_files=args.dataset_name, split="train")
    outs = []
    for one in tqdm(test_examples):
        if "is correct" in one["judge"]:
            outs.append(one)
    
    test_examples = Dataset.from_list(outs)
    test_examples.to_json(args.dataset_name.replace('.json', '_filtered.json'))

if __name__ == '__main__':
    main()
