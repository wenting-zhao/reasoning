import argparse
from copy import deepcopy
import re
import os
import signal
import diskcache as dc
import openai
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import sample_completion, start_server
from compute_accuracy import last_boxed_only_string, remove_boxed

from eval_codegen import check_correctness

import pdb

cache = dc.Cache(os.getcwd() + "/.diskcache")
client = openai.OpenAI()


def format_example(example, include_answer=False):
    prompt = [
        {
            "role": "system",
            "content": f"""You are an expert programmer.
You will be given a problem to solve.

First, list out the steps and helper functions needed to solve the task in the following format:
```plan
1. function1: Type -> Type -> Type. Description.
2. function2: Type -> Type -> Type. Description.
```

Then, give your solution in python:
```python
```""",
        },
        {
            "role": "user",
            "content": f"This is the problem:\n\n{example['question']}"
        }
    ]
    return prompt


def parse_response(text):
    plan_re = r'```plan(.*)```'
    python_re = r'```python(.*)```'
    plan_matches = re.findall(plan_re, text, re.DOTALL)
    python_matches = re.findall(python_re, text, re.DOTALL)
    if len(python_matches) == 0 or len(plan_matches) == 0:
        pdb.set_trace()
    return python_matches[0], plan_matches[0]


# @cached(cache, lambda *args, **kwargs: json.dumps(args) + json.dumps(kwargs))
def prompt(
    messages,
    model="gpt-4o",
    max_tokens=512,
    temperature=1,
    n=1,
):
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        n=n,
    )
    return [x.message.content for x in response.choices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="gpt-4o", help="The name of the model to use."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="codeparrot/apps",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Which split of the dataset to load.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_samples", type=int, default=1, help="number of samples to generate"
    )
    parser.add_argument("--start", type=int, default=0, help="start of the dataset")
    parser.add_argument("--end", type=int, default=1000, help="end of the dataset")
    parser.add_argument(
        "--nofewshot", action="store_true", help="later iterations require no fewshot."
    )
    args = parser.parse_args()

    datasets = load_dataset(
        args.dataset_name, args.dataset_config_name, split=args.dataset_split
    )
    test_examples = datasets.select(range(args.start, args.end))

    outs = []
    for example in tqdm(test_examples):
        answers = prompt(format_example(example), model=args.model_name, n=args.num_samples)
        codes, plans = zip(*map(parse_response, answers))
        import pdb; pdb.set_trace()
        result = check_correctness(example, codes)
        outs.append(answers)

    test_examples = test_examples.add_column(name="model-a", column=outs)
    pdb.set_trace()
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_name.split("/")[-1]
    out_name = f"out/model-a-samples-{dataset_name}-{args.dataset_split}-{model_name}-num{args.num_samples}-start{args.start}-end{args.end}.json"
    test_examples.to_json(out_name)


if __name__ == "__main__":
    main()
