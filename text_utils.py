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

