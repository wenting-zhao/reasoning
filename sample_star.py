import argparse
from copy import deepcopy
import re
import os
import signal
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import sample_completion, start_server 
from compute_accuracy import last_boxed_only_string, remove_boxed 

import sglang as sgl
from sglang.backend.runtime_endpoint import RuntimeEndpoint

def format_example(example, include_answer=True):
    example['answer'] = remove_boxed(last_boxed_only_string(example['solution']))
    prompt = [{"role": "user", "content": f"You will be given a math problem with an answer specified. Your job is to show the derivation to the answer.\n\nProblem: {example['problem']}\nAnswer: {example['answer']}"}]
    if include_answer:
        prompt += [{"role": "assistant", "content": f"{example['solution']}"}]
    return prompt

def gen_prompt(test_example, fewshot_examples):
    prompt = []
    if len(fewshot_examples) > 0:
        for one in fewshot_examples:
            prompt += format_example(one, include_answer=True)
    prompt += format_example(test_example, include_answer=False)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_split", type=str, required=True, default=None, help="Which split of the dataset to load.")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--start", type=int, default=0, help="start of the dataset")
    parser.add_argument("--end", type=int, default=1000, help="end of the dataset")
    parser.add_argument("--port", type=str, default="30000", help="port number")
    parser.add_argument("--nofewshot", action="store_true", help="later iterations require no fewshot.")
    parser.add_argument("--start_server", action="store_true", help="whether to start sglang server")
    args = parser.parse_args()

    if args.start_server:
        pro = start_server(args.model_name, args.port)
    sgl.set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))

    datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    test_examples = datasets[args.dataset_split].select(range(args.start, args.end))
    fewshot_examples = datasets['train'].select([1708, 7098, 1076, 600])
    outs = []
    for i in tqdm(range(0, len(test_examples), args.batch_size)):
        data = test_examples.select(range(i, min(i+args.batch_size, len(test_examples))))
        in_text = []
        for one in data:
            if args.nofewshot:
                in_text.append(gen_prompt(one, []))
            else:
                in_text.append(gen_prompt(one, fewshot_examples))
        answer = sample_completion(in_text, samples=args.num_samples)
        outs += answer
 
    outs = [outs[i:i+args.num_samples] for i in range(0, len(outs), args.num_samples)]
    test_examples = test_examples.add_column(name='star', column=outs)
    dataset_name = args.dataset_name.split('/')[-1]
    model_name = args.model_name.split('/')[-1]
    out_name = f"out/star-samples-{dataset_name}-{args.dataset_split}-{model_name}-num{args.num_samples}-start{args.start}-end{args.end}.json"
    test_examples.to_json(out_name)
    if args.start_server:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == '__main__':
    main()
