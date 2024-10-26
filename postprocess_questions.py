from datasets import load_dataset
import sys

def format_question(text, eos_token="<|eot_id|>"):
    text = text.split("user")[-1]
    start_idx = text.find("\n")
    end_idx = text.rfind(eos_token)
    text = text[start_idx:end_idx].strip()
    #print(text)
    #print("="*10)
    if "Question:" in text:
        text = text.split("Question:")[1].strip()
    if "Problem:" in text:
        text = text.split("Problem:")[1].strip()
    if "?\n" in text:
        text = text.split("?\n")[0].strip()+'?'
    text = text.split("Solution:")[0].strip()
    text = text.split("Final Answer:")[0].strip()
    text = text.split("\nWe know")[0].strip()
    text = text.split("\nLet's")[0].strip()
    idx = text.find("\nFind")
    idx2 = text[idx:].find("\n")
    text = text[:idx+idx2].strip()
    return text

ds = load_dataset("json", data_files=sys.argv[1], split="train")
for one in ds:
    out = format_question(one["text"])
    print(out)
    print("="*10)
