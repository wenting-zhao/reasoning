import os
import subprocess
import sglang as sgl


def start_server(model, port):
    process = subprocess.Popen(["python", "-m", "sglang.launch_server", "--model-path", model, "--port", port])
    os.system("sleep 30s")
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

def sample_completion(text, temperature=1, max_tokens=512, samples=50):
    if isinstance(text, str):
        d = [{"question": text} for _ in range(samples)]
    elif isinstance(text, list) and len(text) == 1:
        d = [{"question": text[0]} for _ in range(samples)]
    elif isinstance(text, list):
        assert samples == 1
        d = [{"question": one} for one in text]
    states = text_qa.run_batch(d, progress_bar=True, max_new_tokens=max_tokens, temperature=temperature)
    for i in range(len(states)):
        states[i] = states[i].text()
    return states

