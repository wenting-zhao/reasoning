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

def load_model_and_tokenizer(name):
    if "gpt" not in name:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = name
        tokenizer = ""
    return model, tokenizer
