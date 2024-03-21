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
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model"
    )

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_not_sample", action="store_true", help="Disable sampling")
    parser.add_argument("--constraint", action="store_true", help="Whether to use constraints")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument(
        "--diversity_penalty", type=float, default=0.0, help="This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if group beam search is enabled."
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_beam_groups", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="If set to int > 0, all ngrams of that size can only occur once.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    set_seed(args)

    ds = load_dataset(args.dataset_name, args.dataset_config_name, split='test')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.to(args.device)

    logger.info(args)

    generated_sequences = []

    for data in tqdm(ds):
        prompt_text = data['problem'] + ' </s> '
        d = tokenizer(prompt_text, return_tensors="pt")
        for key in d:
            d[key] = d[key].to(args.device)

        out = model.generate(
            **d,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=5,
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.do_not_sample,
            num_return_sequences=args.num_return_sequences,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty
        )

        # Remove the batch dimension when returning multiple sequences
        if len(out.shape) > 2:
            out.squeeze_()
        output_sequences = out.cpu().tolist()

        curr_seq = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True).split(' </s> ')[1]
            text = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)[0].strip()
            print(text)
            curr_seq.append(text)
        generated_sequences.append(curr_seq)

    ds = ds.add_column(name="samples", column=generated_sequences)
    dataset_name, split = args.dataset.split('/')[-1]
    split = split.replace(".json", "")
    model_name = args.model_name_or_path.replace('/', '-')
    out_name = f"{dataset_name}-{model_name}-temp-{args.temperature}-k-{args.k}-num-{args.num_return_sequences}.json"
    if args.num_beams > 1:
        out_name = out_name.replace(".json", f"-beam-{args.num_beams}.json")
    if args.no_repeat_ngram_size > 0:
        out_name = out_name.replace(".json", f"-ngram-{args.no_repeat_ngram_size}.json")
    if args.repetition_penalty > 1:
        out_name = out_name.replace(".json", f"-rp-{args.repetition_penalty}.json")
    if args.repetition_penalty != 0:
        out_name = out_name.replace(".json", f"-dp-{args.diversity_penalty}.json")
    if args.num_beam_groups > 1:
        out_name = out_name.replace(".json", f"-bg-{args.num_beam_groups}.json")
    if args.num_return_sequences == 1 and args.do_not_sample:
        out_name = f"greedy-{out_name}"
    ds.to_json(out_name)


if __name__ == "__main__":
    main()
