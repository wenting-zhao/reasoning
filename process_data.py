import sys
import re

from datasets import load_dataset, Dataset

if sys.argv[1] == "gsm8k":
    ds = load_dataset("gsm8k", "socratic")["train"]
    outs = []
    for one in ds:
        s = one['answer'].split("####")[0]
        s = re.sub(r'<<.*>>', '', s)
        s = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', s)
        s = [one.split('\n') for one in s]
        s = [one_i.strip() for one in s for one_i in one if one_i.strip() != ""]
        if len(s) % 2 != 0:
            print("WRONG SPLITTING")
            print("!!!", s)
            continue
        s = [x.replace("**", "").strip() for x in s]
        prefix = one['question']
        for i in range(0, len(s)-1, 2):
            text = prefix.strip() + " " + " ".join(s[:i]) + " </s> " + s[i] 
            text = text.replace("  ", " ")
            outs.append({"text": text})
            print(text)
        print("="*100)
    out_ds = Dataset.from_list(outs)
    out_ds.to_json("data/fine-tuning/gsm8k-questions.json")
else:
    new_ds = []
    ds = load_dataset('json', data_files=sys.argv[1:], split='train')
    for one in ds:
        text = one['output']
        curr = []
        while True:
            q_idx = text.find('Sub-problem')
            a_idx = text.find('Solution')
            q = text[q_idx:a_idx]
            text = text[a_idx:]
            if 'Sub-problem' not in text:
                a = text
                break
            else:
                q_idx = text.find('Sub-problem')
                a = text[:q_idx]
            text = text[q_idx:]
            curr.append([q.replace('Sub-problem:', '').strip(), a.replace('Solution to the sub-problem:', '').strip()])
        if len(curr) > 0:
            one['steps'] = curr
            new_ds.append(one)

    outs = []
    for one in new_ds:
        for i in range(len(one['steps'])):
            first = [' '.join(x) for x in one['steps'][:i]]
            text = ' '.join(first)
            text = text.strip()
            prefix = one['problem']
            text = prefix.strip() + ' ' + text.strip() + ' </s> ' + one['steps'][i][0].strip()
            text = text.replace("  ", " ")
            outs.append({"text": text})
            print(text)
        print("="*100)
    out_ds = Dataset.from_list(outs)
    out_ds.to_json(f"data/fine-tuning/{sys.argv[1].replace('out/', '')}")
