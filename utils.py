import re
import time
import torch
import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def query_chat(messages, model, tokenizer=None, temperature=1, max_tokens=512):
    if isinstance(model, str):
        error = True
        while error:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                output = response.choices[0].message.content.strip()
                error = False
            except openai._exceptions.OpenAIError as e:
                if 'context_length_exceeded' in str(e):
                    if len(messages) > 1:
                        messages = messages[-1:]
                    else:
                        messages[-1]['content'] = messages[-1]['content'][int(0.9*len(messages[-1]['content'])):]
                time.sleep(5)
                print(type(e), e)
    else:
        if 'system' not in tokenizer.chat_template or 'System role not supported' in tokenizer.chat_template:
            messages[1]["content"] = messages[0]["content"] + '\n\n' + messages[1]["content"]
            messages = messages[1:]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        attn_mask = torch.ones_like(tokenized_chat, device=device)
        with torch.no_grad():
            outputs = model.generate(tokenized_chat, attention_mask=attn_mask, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
        output = tokenizer.decode(outputs[0][len(tokenized_chat[0]):], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
    return output

def sample_completion(text, model, tokenizer=None, temperature=1, max_tokens=512, samples=50):
    d = tokenizer(text, return_tensors="pt")
    for key in d:
        d[key] = d[key].to(device)
    with torch.no_grad():
        out = model.generate(
            **d,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=samples,
        )
    # Remove the batch dimension when returning multiple sequences
    if len(out.shape) > 2:
        out.squeeze_()
    output_sequences = out.cpu().tolist()

    curr_seq = []
    dupes = set()
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        total_sequence = text[len(tokenizer.decode(d['input_ids'][0], clean_up_tokenization_spaces=True, skip_special_tokens=True)) :]
        if total_sequence in dupes:
            continue
        dupes.add(total_sequence)
        curr_seq.append(total_sequence)
        print(total_sequence)
        print("="*100)
        curr_seq.append(total_sequence)
    return curr_seq

def load_model_and_tokenizer(name):
    if "gpt" not in name:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        model = model.to(device)
        model = torch.compile(model)
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = name
        tokenizer = ""
    return model, tokenizer

def format_reasoning_input(example):
    if len(example['steps']) > 0:
        first = [' '.join(x) for x in example['steps']]
        text = ' '.join(first)
        text = text.strip()
        prefix = example['problem']
        text = prefix.strip() + ' ' + text.strip() + ' </s> '
    else:
        text = example['problem'].strip() + ' </s> '
    return text

def inference_step(example, model, tokenizer, max_tokens=64, temperature=1):
    prompt_text = format_reasoning_input(example) + ' </s> '
    d = tokenizer(prompt_text, return_tensors="pt")
    for key in d:
        d[key] = d[key].to(device)

    outputs = model.generate(
        **d,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )

    # Decode text
    text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)
    idx = text.find('</s>')
    text = text[idx:].replace('</s>', '').strip()
    text = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)[0].strip()
    return text
