import sys

from datasets import load_dataset, Dataset


ds = load_dataset('json', data_files=sys.argv[1], split='train')
new_ds = []
for one in ds:
    text = one['output']
    curr = []
    while True:
        q_idx = text.find('Sub-question')
        a_idx = text.find('Answer')
        q = text[q_idx:a_idx]
        text = text[a_idx:]
        if 'Sub-question' not in text:
            a = text
            break
        else:
            q_idx = text.find('Sub-question')
            a = text[:q_idx]
        text = text[q_idx:]
        curr.append([q.replace('Sub-question:', '').strip(), a.replace('Answer:', '').strip()])
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
        outs.append({"text": text})
        print(text)
    print("="*100)
out_ds = Dataset.from_list(outs)
out_ds.to_json(f"data/fine-tuning/{sys.argv[1].replace('out/', '')}")
