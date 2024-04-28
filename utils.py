import os
import subprocess
import openai

import sglang as sgl

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def start_server(model, port):
    process = subprocess.Popen(["python", "-m", "sglang.launch_server", "--model-path", model, "--port", port])
    os.system("sleep 1m")
    return process

@sgl.function
def text_qa(s, question):
    s += question
    s += sgl.gen("answer")

def sample_completion(text, temperature=1, max_tokens=512, samples=50, stop=None):
    if isinstance(text, str):
        d = [{"question": text} for _ in range(samples)]
    elif isinstance(text, list) and len(text) == 1:
        d = [{"question": text[0]} for _ in range(samples)]
    elif isinstance(text, list):
        assert samples == 1
        d = [{"question": one} for one in text]
    states = text_qa.run_batch(d, progress_bar=True, max_new_tokens=max_tokens, stop=stop, temperature=temperature)
    for i in range(len(states)):
        states[i] = states[i].text()[len(d[i]["question"]):]
    return states

