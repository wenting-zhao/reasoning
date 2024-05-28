import os
import subprocess
import sglang as sgl
import numpy as np


def start_server(model, port):
    process = subprocess.Popen(["python", "-m", "sglang.launch_server", "--model-path", model, "--port", port])
    os.system("sleep 2m")
    return process

@sgl.function
def text_qa(s, question):
    for one in question[:-1]:
        if one["role"] == "user":
            s += sgl.user(one["content"])
        else:
            s += sgl.assistant(one["content"])
    s += sgl.user(question[-1]["content"])
    s += sgl.assistant(sgl.gen("answer"))

def sample_completion(text, temperature=1, max_tokens=1024, samples=50):
    if isinstance(text, str):
        d = [{"question": text} for _ in range(samples)]
    elif isinstance(text, list):
        d = [[{"question": one} for _ in range(samples)] for one in text]
        d = [xx for x in d for xx in x]
    else:
        raise ValueError("input must be either str or list.")
    states = text_qa.run_batch(d, progress_bar=True, max_new_tokens=max_tokens, temperature=temperature)
    for i in range(len(states)):
        states[i] = states[i].text()
    return states


@sgl.function
def codegen(s, system_message, user_message):
    s += sgl.system(system_message)
    s += sgl.user(user_message)
    s += sgl.assistant(sgl.gen("answer"))

def convert_messages(batch):
    return [
        {
            "system_message": example[0]["content"],
            "user_message": example[1]["content"],
        }
        for example in batch
    ]

def sample_code_completion(batch, temperature=1, max_tokens=2048, samples=50):
    states = codegen.run_batch(convert_messages(batch), progress_bar=True, max_new_tokens=max_tokens, temperature=temperature)
    batch_size = len(batch)
    return np.array([s["answer"] for s in states]).reshape(batch_size, samples)

