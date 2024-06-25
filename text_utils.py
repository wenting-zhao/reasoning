import nltk


def format_qa(text, eos_token="<|eot_id|>"):
    text = text.split("assistant")[-1]
    start_idx = text.find("Sub-problem")
    end_idx = text.rfind(eos_token)
    text = text[start_idx:end_idx]
    curr = []
    while True:
        q_idx = text.find('Sub-problem')
        a_idx = text.find('Solution')
        q = text[q_idx:a_idx]
        text = text[a_idx:]
        if 'Sub-problem' not in text:
            a = text
            curr.append([q.replace('Sub-problem:', '').strip(), a.replace('Solution to the sub-problem:', '').strip()])
            break
        else:
            q_idx = text.find('Sub-problem')
            a = text[:q_idx]
            text = text[q_idx:]
            curr.append([q.replace('Sub-problem:', '').strip(), a.replace('Solution to the sub-problem:', '').strip()])
    return curr

def format_cot(text, eos_token="<|eot_id|>"):
    text = text.split("assistant")[-1]
    start_idx = text.find("\n")
    end_idx = text.rfind(eos_token)
    text = text[start_idx:end_idx].strip()
    return text

def split_solutions(text):
    sep1 = "<|start_header_id|>assistant<|end_header_id|>"
    sep2 = "<|eot_id|>"
    outs = []
    while True:
        start = text.find(sep1)
        if start == -1:
            break
        end = text[start:].find(sep2)
        if end == -1:
            break
        end += start
        out = text[start:end+len(sep2)]
        outs.append(out)
        text = text[end+len(sep2):]
    for i in range(1, len(outs)):
        out = outs[i].split("\n\n")

        if out[1].startswith("Here") or "another" in out[1].lower() or "different" in out[1].lower():
            sentences = nltk.sent_tokenize(out[1])
            sentences = sentences[1:]
            out[1] = " ".join(sentences)

        if out[1].endswith("!"):
            out.pop(1)
        outs[i] = '\n\n'.join(out)
    return outs
