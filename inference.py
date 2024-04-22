#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import itertools
import logging
import sys
import re

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils import load_model_and_tokenizer, sample_completion


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
    args = parser.parse_args()

    set_seed(args)

    ds = load_dataset('json', data_files=args.dataset_name, split='train')
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    logger.info(args)

    generated_sequences = []

    for i in tqdm(range(0, len(ds), args.batch_size)):
        data = ds.select(range(i, min(i+args.batch_size, len(ds))))
        prompt_text = [one["text"] for one in data]

        results = sample_completion(prompt_text, model, tokenizer, temperature=args.temperature, max_tokens=args.max_new_tokens, samples=1)
        print(results)
        generated_sequences += results

    ds = ds.add_column(name="output", column=generated_sequences)
    dataset_name = args.dataset_name.split('/')[-1]
    model_name = args.model_name.replace('/', '-')
    out_name = f"out/baselines/{dataset_name}-{model_name}-temp-{args.temperature}.json"
    ds.to_json(out_name)


if __name__ == "__main__":
    main()
