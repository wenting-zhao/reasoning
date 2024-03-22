import argparse
from copy import deepcopy
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import query_chat, load_model_and_tokenizer

fewshot_examples=[{"problem":"What is the number of units in the distance between $(2,5)$ and $(-6,-1)$?","steps":[["How do you find the distance between two points in a coordinate plane?","The distance formula is $\\sqrt{(x2 - x1)^2 + (y2 - y1)^2}$."], ["What is the difference in x-coordinates?","$-6 - 2 = -8$."],["What is the difference in y-coordinates?","$-1 - 5 = -6$."],["Substitute the differences into the distance formula.","$\\sqrt{(-8)^2 + (-6)^2} = \\sqrt{64 + 36} = \\sqrt{100}$."],["What is the square root of 100?","$\\sqrt{100} = 10$"]],"answer":"10"},{"problem":"Find the $\\emph{positive}$ real number(s) $x$ such that $\\frac{1}{2}\\left( 3x^2-1\\right) = \\left( x^2-50x-10\\right)\\left( x^2+25x+5\\right)$.","steps":[["Can you apply substitutions for the equation so that it becomes a linear equation?", "Write $a = x^2-50x-10$ and $b = x^2+25x+5$."],["Substitute the equation with a and b.","the equation given becomes\n\\[\\frac{a+2b-1}{2} = ab,\\] so $0=2ab-a-2b+1=(a-1)(2b-1)$."],["What do we know about $x$ from $(a-1)(2b-1)=0$?","$a-1=x^2-50x-11=0$ or $2b-1=2x^2+50x+9=0$. The former has a positive root, $x=25 + 2\\sqrt{159}$, while the latter does not."]],"answer":"25 + 2\\sqrt{159}"},{"problem":"It took $4$ days for $75$ workers, all working together at the same rate, to build an embankment. If only $50$ workers had been available, how many total days would it have taken to build the embankment?","steps":[["How many total worker-days were required to build the embankment?","$75$ workers working for $4$ days would result in $75*4 = 300$ total worker-days."],["If only $50$ workers were available, how many days would it take to build the embankment?","If $75$ workers completed the project in $4$ days, then $50$ workers would need $300\/50 = 6$ days to complete the same project."]],"answer":"6"},{"problem":"At a particular school with 43 students, each student takes chemistry, biology, or both. The chemistry class is three times as large as the biology class, and 5 students are taking both classes. How many people are in the chemistry class?","steps":[["Define variables for the number of students only in the biology class and only in the chemistry class.","Let $x$ be the number of students in the biology class who aren't in the chemistry class and $y$ be the number of students in the chemistry class who aren't in the biology class."],["Can we express the total number of students using x and y?","Since all students are in either one of the classes or in both, we know that $43=x+y+5$."],["Can you use the chemistry class being three times as large as the biology class to build another equation?","$3(x+5)=y+5$."],["Can you solve for $y$ in terms of $x$?","$y=3x+10$"],["Substitute $y=3x+10$ into the equation $43=x+(3x+10)+5$ to get $x$ and $y$.","$x=7$, and $y=31$"],["Add the number of chemistry students who aren't taking biology to the number of students taking both classes.","$31+5=36$."]],"answer":"36"}]

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

def format_example(example, include_answer=True):
    if include_answer:
        prompt = "Problem: " + example['problem'] + f"\nSolution: {example['answer']}\n\n"
        for q, a in example['steps']:
            prompt += f"Sub-problem: {q}\nSolution to the sub-problem: {a}\n"
        prompt += "\n"
        return prompt
    if 'choices' in example:
        prompt = "Problem: " + example['problem'] + " Which of the following answer choices is correct?"
        choices = ["A", "B", "C", "D"]
        assert len(example['choices']) == len(choices)
        for j, item in enumerate(example['choices']):
            prompt += "\n{}. {}".format(choices[j], item)
        prompt += f"\nSolution: {choices[example['answer']]}. {example['choices'][example['answer']]}\n\n"
    else:
        prompt = "Problem: " + example['problem'] + f"\nSolution: {example['answer']}\n\n"
    return prompt

def gen_prompt(test_example, fewshot_examples):
    prompt = "You will be given an extremely difficult problem with an answer specified. Your job is to show the derivation to the answer. Specifically, your derivation should be structured as a list of sub-problems and solutions to the sub-problems, which will eventually lead to the specified answer.\n\n"
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
    datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    test_examples = datasets[args.dataset_split]
    count = 0
    n_correct = 0
    outs = []
    for one in tqdm(test_examples):
        if "mmlu" in args.dataset_name and '|' in one['question']:
            continue
        if "competition_math" in args.dataset_name:
            one['answer'] = extract_substrings(one['solution'])
        if 'problem' not in one:
            one['problem'] = deepcopy(one['question'])
            del one['question']
        in_text = gen_prompt(one, fewshot_examples)
        gpt_answer = query_chat([{'role': 'user', 'content': in_text}], model=model)
        one['output'] = gpt_answer
        outs.append(one)
    
    test_examples = Dataset.from_list(outs)
    dataset_name = args.dataset_name.split('/')[-1]
    test_examples.to_json(f"out/{dataset_name}-{args.dataset_split}-questions-{args.model_name}.json")

if __name__ == '__main__':
    main()
