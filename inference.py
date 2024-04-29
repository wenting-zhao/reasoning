import argparse
import itertools
import logging
import sys
import re
import os
import signal

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import sample_completion, start_server

import sglang as sgl
from sglang.backend.runtime_endpoint import RuntimeEndpoint


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model"
    )

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_samples", type=int, default=1, help="how many samples to generate")
    parser.add_argument("--port", type=str, default="30000", help="port number")
    parser.add_argument("--start_server", action="store_true", help="whether to start sglang server")
    args = parser.parse_args()

    set_seed(args)

    ds = load_dataset('json', data_files=args.dataset_name, split='train')

    if args.start_server:
        pro = start_server(args.model_name, args.port)
    sgl.set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))

    logger.info(args)

    generated_sequences = []

    for i in tqdm(range(0, len(ds), args.batch_size)):
        data = ds.select(range(i, min(i+args.batch_size, len(ds))))
        prompt_text = [one["text"][:-1] for one in data]

        results = sample_completion(prompt_text, temperature=args.temperature, max_tokens=args.max_new_tokens, samples=args.num_samples)
        generated_sequences += results

    generated_sequences = [generated_sequences[i:i+args.num_samples] for i in range(0, len(generated_sequences), args.num_samples)]
    ds = ds.add_column(name="output", column=generated_sequences)
    dataset_name = args.dataset_name.split('/')[-1].replace(".json", "")
    model_name = args.model_name.split('/')[-1]
    out_name = f"out/inference/{dataset_name}-{model_name}-num{args.num_samples}.json"
    ds.to_json(out_name)
    if args.start_server:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()
