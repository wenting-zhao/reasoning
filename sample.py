import argparse
from copy import deepcopy
import re
import os
import signal
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import sample_completion, start_server 
from compute_accuracy import last_boxed_only_string, remove_boxed 
from sample_model_a import gen_prompt as gen_prompt_a
from sample_model_b import gen_prompt as gen_prompt_b

import sglang as sgl
from sglang.backend.runtime_endpoint import RuntimeEndpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_name", type=str, required=True, help="The name of the model a to use.")
    parser.add_argument("--model_b_name", type=str, required=True, help="The name of the model b to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_split", type=str, required=True, default=None, help="Which split of the dataset to load.")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--start", type=int, default=0, help="start of the dataset")
    parser.add_argument("--end", type=int, default=1000, help="end of the dataset")
    parser.add_argument("--port", type=str, default="30000", help="port number")
    parser.add_argument("--nofewshot", action="store_true", help="later iterations require no fewshot.")
    args = parser.parse_args()

    datasets = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split)
    test_examples = datasets.select(range(args.start, args.end))

    pro = start_server(args.model_a_name, args.port)
    print(pro.pid)
    sgl.set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))
    outs = []
    for i in tqdm(range(0, len(test_examples), args.batch_size)):
        data = test_examples.select(range(i, min(i+args.batch_size, len(test_examples))))
        in_text = []
        for one in data:
            if "competition_math" in args.dataset_name:
                one['answer'] = remove_boxed(last_boxed_only_string(one['solution']))
            in_text.append(gen_prompt_a(one, not args.nofewshot))
        answer = sample_completion(in_text, samples=args.num_samples)
        outs += answer 
    outs = [outs[i:i+args.num_samples] for i in range(0, len(outs), args.num_samples)]
    test_examples = test_examples.add_column(name='model-a', column=outs)
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

    pro = start_server(args.model_b_name, args.port)
    sgl.set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))
    outs = []
    for i in tqdm(range(0, len(test_examples), args.batch_size)):
        data = test_examples.select(range(i, min(i+args.batch_size, len(test_examples))))
        in_text = []
        for one in data:
            in_text += gen_prompt_b(one, not args.nofewshot)
        answer = sample_completion(in_text, samples=1)
        outs += answer
    outs = [outs[i:i+len(data[0]['model-a'])] for i in range(0, len(outs), len(data[0]['model-a']))]
    test_examples = test_examples.add_column(name='model-b', column=outs)

    dataset_name = args.dataset_name.split('/')[-1]
    model_name = args.model_name.split('/')[-1]
    out_name = f"out/game-samples-{dataset_name}-{args.dataset_split}-{model_name}-num{args.num_samples}-start{args.start}-end{args.end}.json"
    test_examples.to_json(out_name)
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == '__main__':
    main()
