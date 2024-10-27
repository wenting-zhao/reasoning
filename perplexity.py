import argparse
import sys
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from dataclasses import dataclass, field

import evaluate
from evaluate import logging
from datasets import load_dataset

from tqdm import tqdm
import pdb
from text_utils import format_cot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model_name_or_path = sys.argv[1]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=sys.argv[2], split="train")

    def tokenize_function(examples):
        key = 'output' if 'output' in examples else 'star'
        #inp = [[{"role": "user", "content": "Solve the following math problem.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n" + problem}, {"role": "assistant", "content": format_cot(x)}] for problem, outputs in zip(examples["problem"], examples[key]) for x in outputs]
        inp = [{"role": "user", "content": f"You will be given a math problem with an answer specified. Your job is to show the derivation to the answer.\n\nProblem: {example['problem']}\nAnswer: {example['answer']}"}, {"role": "assistant", "content": format_cot(x)}]
        inp = [tokenizer.apply_chat_template(x, tokenize=False) for x in inp]
        output = tokenizer(inp)
        return output

    ds = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )
    tokenizer.padding_side = "right"

    loss_fct = CrossEntropyLoss(reduction="none")
    dataloader = DataLoader(
        ds, collate_fn=data_collator, batch_size=4
    )
    mean_logprobs = []
    for batch in tqdm(dataloader):
        for key in batch:
            batch[key] = batch[key].to(device)
        labels = batch["input_ids"].clone().detach()
        indices = (batch["input_ids"] == tokenizer.eos_token_id).cumsum(dim=1) == 0
        labels[indices] = -100
        labels[batch["input_ids"]==tokenizer.eos_token_id] = -100

        with torch.no_grad():
            out_logits = model(**batch, labels=labels).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logprob = -loss_fct(shift_logits.transpose(1, 2), shift_labels)
        l = torch.count_nonzero(logprob, dim=-1)
        logprob = logprob.sum(1)
        mean_logprob = logprob / l

        mean_logprobs += mean_logprob.tolist()
    ds = load_dataset("json", data_files=sys.argv[2], split="train")
    key = 'output' if 'output' in ds.column_names else 'star'
    n = len(ds[key][0])
    mean_logprobs = [mean_logprobs[i:i + n] for i in range(0, len(mean_logprobs), n)]
    ds = ds.add_column(name="logprob", column=mean_logprobs)
    ds.to_json(sys.argv[2].replace(".json", "_logprob.json"))


if __name__ == "__main__":
    main()
