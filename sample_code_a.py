import argparse
from copy import deepcopy
import re
import os
import signal
import json
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import sample_code_completion, start_server
from compute_accuracy import last_boxed_only_string, remove_boxed

import sglang as sgl
from sglang.backend.runtime_endpoint import RuntimeEndpoint

from eval_codegen import run_test, ErrorType, get_error_type

import pdb

# initial
fewshot_idxs = [(21,1), (32,0)]
# more examples with len(plans) > 2
fewshot_idxs = [(3,0), (5,0), (8,3), (11,0), (21,1), (22,1), (32,0)]

# old prompts with planning
'''
def format_fewshot_example(example, attempt):
    return f"""# Example Problem
{example["question"]}

# Plan
{example["plan"][attempt]}

# Solution
{example["code"][attempt]}"""


ZEROSHOT_STRING = """You are an expert programmer.
You will be given a problem to solve.

First, list out the steps and helper functions needed to solve the task in the following format:
# Plan
1. function1: Type -> Type -> Type. Description.
2. function2: Type -> Type -> Type. Description.

# Solution
```python
# code solution here
```"""
'''

# new prompts without planning
def format_fewshot_example(example):
    return f"""# Example Problem
{example["question"].strip()}

# Example Solution
```python
{json.loads(example["solutions"])[0].strip()}
```"""


ZEROSHOT_STRING = """You are an expert programmer.
You will be given a problem to solve. Please write code for a solution.

The problem will require you to read input from stdin, and print the solution to stdout.

# Solution
```python
# code solution here
```"""


NEWLINE = "\n\n"

def format_example(example, fewshot_examples=None):
    prompt = [
        {
            "role": "system",
            "content": ZEROSHOT_STRING
                if fewshot_examples is None
                else f"{ZEROSHOT_STRING}\n{NEWLINE.join(fewshot_examples)}",
        },
        {
            "role": "user",
            "content": f"This is the problem:\n\n# Problem\n{example['question']}\n# Solution"
        } 
    ]
    return prompt


def parse_response(text):
    #plan_re = r'```plan(.*)```'
    #plan_re = r'# Plan\n(.*)# Solution'
    #python_re = r'```python(.*)```'
    python_re = r'```(?:python)?(.*)```'
    #plan_matches = re.findall(plan_re, text, re.DOTALL)
    python_matches = re.findall(python_re, text, re.DOTALL)

    # generation failure => parsing failure
    #if len(plan_matches) == 0:
        #plan_matches = [""]
    if len(python_matches) == 0:
        python_matches = [""]

    #return python_matches[0], plan_matches[0]
    return python_matches[0], None


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
    parser.add_argument("--use_sglang", action="store_true", help="use sglang")
    parser.add_argument("--start_server", action="store_true", help="whether to start sglang server")
    parser.add_argument("--port", type=str, default="30000", help="port number")
    args = parser.parse_args()

    if args.use_sglang:
        # override port
        port = os.getenv("SGLANG_PORT") or args.port
        print(f"Using SGLANG on port {port}")

    if args.start_server:
        pro = start_server(args.model_name, args.port)
    if args.use_sglang:
        sgl.set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
    else:
        sgl.set_default_backend(sgl.OpenAI("gpt-4o"))

    datasets = load_dataset(
        args.dataset_name, args.dataset_config_name, split=args.dataset_split
    )
    test_examples = datasets.select(range(args.start, args.end))

    fewshot_examples = None
    if not args.nofewshot:
        '''
        annotated_dataset = load_dataset(
            "json",
            data_files="out/model-a-samples-apps-train-gpt-4o-num8-start0-end128.json",
        )
        fewshot_examples = [
            format_fewshot_example(annotated_dataset["train"][idx], attempt)
            for idx, attempt in fewshot_idxs
        ]
        '''
        train = load_dataset(args.dataset_name, split="train")
        fewshot_examples = [
            format_fewshot_example(x) for x in train.select(range(4))
        ]


    print("RUNNING GENERATION")
    plan_outputs = []
    code_outputs = []
    for i in tqdm(range(0, len(test_examples), args.batch_size)):
        batch = test_examples.select(range(i, min(i+args.batch_size, len(test_examples))))
        examples = [format_example(example, fewshot_examples) for example in batch]
        answers = sample_code_completion(examples, samples=args.num_samples)
        batch_codes, batch_plans = np.vectorize(parse_response)(answers)
        code_outputs.append(batch_codes)

    #flat_plan_outputs = np.concatenate(plan_outputs, axis=0).tolist()
    flat_code_outputs = np.concatenate(code_outputs, axis=0).tolist()

    test_examples = test_examples.add_column(name="code", column=flat_code_outputs)
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_name.split("/")[-1]
    out_name = f"out/model-a-samples-{dataset_name}-{args.dataset_split}-{model_name}-nofs{args.nofewshot}-num{args.num_samples}-start{args.start}-end{args.end}.json"
    test_examples.to_json(out_name)
    print(f"Saved generations to {out_name}")

    print("RUNNING EVALUATION")
    is_correct = []
    errors = []
    for batch_codes in code_outputs:
        results = [
            [
                result
                for code in codes
                for result in run_test(example, code)
            ]
            for example, codes in zip(batch, batch_codes)
        ]
        # result can be
        # # True if correct
        # * False if incorrect
        # * -1 if timeout
        # * -2 if compilation error (whatever that means for python?)
        #plan_outputs.append(batch_plans)
        is_correct.append(np.vectorize(lambda x: x == True)(results))
        errors.append(np.vectorize(get_error_type)(results))

    flat_is_correct  = np.concatenate(is_correct, axis=0).tolist()
    flat_errors = np.concatenate(errors, axis=0).tolist()

    #test_examples = test_examples.add_column(name="plan", column=plan_outputs)
    test_examples = test_examples.add_column(name="is_correct", column=flat_is_correct)
    out_name = f"out/model-a-samples-{dataset_name}-{args.dataset_split}-{model_name}-nofs{args.nofewshot}-num{args.num_samples}-start{args.start}-end{args.end}-eval.json"
    test_examples.to_json(out_name)
    print(f"Added eval results to {out_name}")


if __name__ == "__main__":
    main()
