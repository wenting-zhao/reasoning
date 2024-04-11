import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
from utils import query_chat, load_model_and_tokenizer


def extract_substrings(text):
    parts = text.split(r"\boxed")
    matches = []
    for part in parts[1:]:  # Skip the first part as it does not start with \boxed
        if part.startswith("{"):
            brace_level = 0
            for i, char in enumerate(part):
                if char == "{":
                    brace_level += 1
                elif char == "}":
                    brace_level -= 1
                    if brace_level == 0:
                        matches.append(
                            part[1:i]
                        )  # Extract the content inside the braces
                        break

    if len(matches) == 0:
        return None

    return matches[0]

def check_answer(official, student):
    return abs(official - student) < (abs(official) + 1e-6) * 1e-6

def extract_and_convert_fraction(text):
    pattern = r"^\\frac\{(\d+)\}\{(\d+)\}$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return float(numerator) / float(denominator)

    pattern = r"^\\frac(\d)(\d)$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return float(numerator) / float(denominator)

    pattern = r"^-\\frac\{(\d+)\}\{(\d+)\}$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return -float(numerator) / float(denominator)

    pattern = r"^-\\frac(\d)(\d)$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return -float(numerator) / float(denominator)
    return text

def remove_latex_text_commands(text):
    """
    Removes all occurrences of \text{...} from a given LaTeX string.

    Parameters:
    text (str): The input LaTeX string.

    Returns:
    str: The LaTeX string with all \text{...} commands removed.
    """
    # Regular expression pattern to match \text{...}
    pattern = r"\\text\{.*?\}"

    # Replace all occurrences of \text{...} with an empty string
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned_text

def format_example(example, include_answer=True):
    prompt = "Problem: " + example['problem'] + "\nSolution:"
    answer = extract_substrings(example["solution"])
    if include_answer:
        prompt += f" {example['solution']}\n\n"
    return prompt

def gen_prompt(test_example, fewshot_examples):
    prompt = "Solve the following math problems.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n"
    for one in fewshot_examples:
        prompt += format_example(one)
    prompt += format_example(test_example, include_answer=False)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_split", type=str, default=None, help="Which split of the dataset to load.")
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer(args.model_name)
    if args.dataset_name.endswith('json'):
        test_examples = load_dataset('json', data_files=args.dataset_name, split="train")
    else:
        datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        test_examples = datasets[args.dataset_split].shuffle(seed=42).select(range(500))
        fewshot_examples = datasets['train'].shuffle(seed=42).select(range(4))
    n_correct = 0
    outs = []
    for one in tqdm(test_examples):
        if 'output' in one:
            gpt_answer = one['output']
        else:
            in_text = gen_prompt(one, fewshot_examples)
            gpt_answer = query_chat([{'role': 'user', 'content': in_text}], model=model)
        outs.append(gpt_answer)

        official_answer = (
            extract_substrings(one["solution"])
            .replace(" ", "")
            .replace("dfrac", "frac")
        )
        if official_answer.startswith(r"\$"):
            official_answer = official_answer[2:]

        try:
            gpt_answer = extract_substrings(gpt_answer)
            gpt_answer = gpt_answer.replace(" ", "").replace("dfrac", "frac")
            official_answer = remove_latex_text_commands(official_answer).replace(
                " ", ""
            )
            gpt_answer = remove_latex_text_commands(gpt_answer).replace(" ", "")
            official_answer = extract_and_convert_fraction(official_answer)
            gpt_answer = extract_and_convert_fraction(gpt_answer)

            if gpt_answer == official_answer:
                n_correct += 1
                continue

            official_float = float(official_answer)
            gpt_float = float(gpt_answer)
            n_correct += check_answer(official_float, gpt_float)
        except Exception as e:
            print("=" * 80 + "\n")
            print("official_answer:" + str(official_answer) + "\n")
            print("gpt_answer:" + str(gpt_answer) + "\n")
            print("-" * 40 + "\n")
            print(one["solution"] + "\n")
            print("-" * 40 + "\n")
            print(outs[-1] + "\n")

    print(
        "n_correct:",
        n_correct,
        "n_total:",
        len(test_examples),
        "accuracy:",
        n_correct / len(test_examples),
    )
    
    if 'output' not in test_examples.column_names:
        test_examples = test_examples.add_column(name='output', column=outs)
        dataset_name = args.dataset_name.split('/')[-1]
        #test_examples.to_json(f"out/small-{dataset_name}-outputs-{args.model_name}.json")
        test_examples.to_json(f"out/small-{dataset_name}-outputs-{args.model_name}-beam.json")

if __name__ == '__main__':
    main()
