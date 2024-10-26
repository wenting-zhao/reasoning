import argparse
import itertools
import logging
import sys
import re
import os
import signal

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from utils import sample_question, start_server

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
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model"
    )

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_samples", type=int, default=32, help="how many samples to generate")
    parser.add_argument("--total", type=int, default=5000, help="how many iterations")
    parser.add_argument("--port", type=str, default="30000", help="port number")
    parser.add_argument("--start_server", action="store_true", help="whether to start sglang server")
    args = parser.parse_args()

    set_seed(args)

    if args.start_server:
        pro = start_server(args.model_name, args.port)
    sgl.set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))

    logger.info(args)

    prompt_text = "Solve the following competition-level math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit."
    generated_sequences = []

    for i in tqdm(range(args.total)):
        results = sample_question(prompt_text, temperature=args.temperature, max_tokens=args.max_new_tokens, samples=args.num_samples)
        generated_sequences += results
    generated_sequences = [{"text": x} for x in generated_sequences]
    ds = Dataset.from_list(generated_sequences)
    ds.to_json(f"out/ehr_{id(ds)}.json")

    if args.start_server:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()
