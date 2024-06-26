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

from reasoning.eval.eval_codegen import run_test, ErrorType, get_error_type
from reasoning.prompt.codegen import AppsStdInPrompt, AppsCallPrompt

import pdb


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

    stdin_prompter = AppsStdInPrompt(use_fewshot=not args.nofewshot, k=4)
    call_prompter = AppsCallPrompt(use_fewshot=not args.nofewshot, k=4)

    def format_example(example):
        return (
            stdin_prompter.render(example)
            if example["starter_code"] == ""
            else call_prompter.render(example)
        )

    print("RUNNING GENERATION")
    plan_outputs = []
    code_outputs = []
    for i in tqdm(range(0, len(test_examples), args.batch_size)):
        batch = test_examples.select(range(i, min(i+args.batch_size, len(test_examples))))
        examples = [format_example(example) for example in batch]
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
