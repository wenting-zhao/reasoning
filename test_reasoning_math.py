import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
from utils import query_chat, load_model_and_tokenizer
from utils import inference_step

fewshot_examples=[{"problem":"What is the number of units in the distance between $(2,5)$ and $(-6,-1)$?","steps":[["How do you find the distance between two points in a coordinate plane?","The distance formula is $\\sqrt{(x2 - x1)^2 + (y2 - y1)^2}$."], ["What is the difference in x-coordinates?","$-6 - 2 = -8$."],["What is the difference in y-coordinates?","$-1 - 5 = -6$."],["Substitute the differences into the distance formula.","$\\sqrt{(-8)^2 + (-6)^2} = \\sqrt{64 + 36} = \\sqrt{100}$."],["What is the square root of 100?","$\\sqrt{100} = 10$"]],"answer":"\\boxed{10}"},{"problem":"Find the $\\emph{positive}$ real number(s) $x$ such that $\\frac{1}{2}\\left( 3x^2-1\\right) = \\left( x^2-50x-10\\right)\\left( x^2+25x+5\\right)$.","steps":[["Can you apply substitutions for the equation so that it becomes a linear equation?", "Write $a = x^2-50x-10$ and $b = x^2+25x+5$."],["Substitute the equation with a and b.","the equation given becomes\n\\[\\frac{a+2b-1}{2} = ab,\\] so $0=2ab-a-2b+1=(a-1)(2b-1)$."],["What do we know about $x$ from $(a-1)(2b-1)=0$?","$a-1=x^2-50x-11=0$ or $2b-1=2x^2+50x+9=0$. The former has a positive root, $x=\\boxed{25 + 2\\sqrt{159}}$, while the latter does not."]],"answer":"\\boxed{25 + 2\\sqrt{159}}"},{"problem":"It took $4$ days for $75$ workers, all working together at the same rate, to build an embankment. If only $50$ workers had been available, how many total days would it have taken to build the embankment?","steps":[["How many total worker-days were required to build the embankment?","$75$ workers working for $4$ days would result in $75*4 = 300$ total worker-days."],["If only $50$ workers were available, how many days would it take to build the embankment?","If $75$ workers completed the project in $4$ days, then $50$ workers would need $300\/50 = 6$ days to complete the same project."]],"answer":"\\boxed{6}"},{"problem":"At a particular school with 43 students, each student takes chemistry, biology, or both. The chemistry class is three times as large as the biology class, and 5 students are taking both classes. How many people are in the chemistry class?","steps":[["Define variables for the number of students only in the biology class and only in the chemistry class.","Let $x$ be the number of students in the biology class who aren't in the chemistry class and $y$ be the number of students in the chemistry class who aren't in the biology class."],["Can we express the total number of students using x and y?","Since all students are in either one of the classes or in both, we know that $43=x+y+5$."],["Can you use the chemistry class being three times as large as the biology class to build another equation?","$3(x+5)=y+5$."],["Can you solve for $y$ in terms of $x$?","$y=3x+10$"],["Substitute $y=3x+10$ into the equation $43=x+(3x+10)+5$ to get $x$ and $y$.","$x=7$, and $y=31$"],["Add the number of chemistry students who aren't taking biology to the number of students taking both classes.","$31+5=\\boxed{36}$."]],"answer":"\\boxed{36}"}]

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
    prompt = "Problem: " + example['problem'] + "\n"
    if include_answer:
        for q, a in example['steps']:
            prompt += f"Sub-question: {q}\nSolution: {a}\n"
        prompt += "Answer: \\boxed{"+example["answer"]+"}"
        prompt += "\n\n"
    elif len(example['steps']) > 0:
        for one in example['steps']:
            if len(one) == 1:
                q = one[0]
                prompt += f"Sub-question: {q}"
            else:
                q, a = one
                prompt += f"Sub-question: {q}\nSolution: {a}\n"
    return prompt

def gen_prompt(test_example, fewshot_examples):
    prompt = "Solve the following math problems by decomposing them into sub-problems.\nPlease highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n"
    for one in fewshot_examples:
        prompt += format_example(one)
    prompt += format_example(test_example, include_answer=False)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--reasoning_model_name", type=str, required=True, help="The name of the reasoning model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_split", type=str, default=None, help="Which split of the dataset to load.")
    parser.add_argument("--max_retry", type=int, default=10, help="max number of times to retry")
    args = parser.parse_args()

    model, _ = load_model_and_tokenizer(args.model_name)
    reasoning_model, tok = load_model_and_tokenizer(args.reasoning_model_name)
    datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    test_examples = datasets[args.dataset_split].shuffle(seed=42).select(range(500))
    n_correct = 0
    outs = []
    for one in tqdm(test_examples):
        one['steps'] = []
        print(one['problem'])
        num_retry = 0
        while True and num_retry < args.max_retry:
            num_retry += 1
            while True: 
                question = inference_step(one, reasoning_model, tok)
                if question != '':
                    break
            one['steps'].append([question])
            in_text = gen_prompt(one, fewshot_examples)
            curr = query_chat([{'role': 'user', 'content': in_text}], model=model)
            idx = curr.find("Sub-question")
            if idx > 0:
                curr = curr[:idx]
            curr = curr.replace('Solution: ', '').strip()
            one['steps'][-1].append(curr)
            res = extract_substrings(curr)
            print("---STEPS---", one['steps'][-1])
            if res is not None:
                gpt_answer = curr
                print("FOUND SOLUTION:", res)
                print("GOLD  SOLUTION:", one["solution"])
                print("="*100)
                break
        else:
            gpt_answer = curr
        outs.append(one['steps'])

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
            print(outs[-1][-1], "\n")

    print(
        "n_correct:",
        n_correct,
        "n_total:",
        len(test_examples),
        "accuracy:",
        n_correct / len(test_examples),
    )
    
    test_examples = test_examples.add_column(name='output', column=outs)
    test_examples.to_json(f"out/small-reasoning-math-outputs-{args.model_name}-{args.reasoning_model_name.split('/')[-1]}.json")

if __name__ == '__main__':
    main()
