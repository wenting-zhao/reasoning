import argparse
from copy import deepcopy
import re
import os
import signal
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import sample_completion, start_server 

import sglang as sgl
from sglang.backend.runtime_endpoint import RuntimeEndpoint

fewshot_examples=[{"problem":"What is the number of units in the distance between $(2,5)$ and $(-6,-1)$?","steps":[["How do you find the distance between two points in a coordinate plane?","The distance formula is $\\sqrt{(x2 - x1)^2 + (y2 - y1)^2}$."], ["What is the difference in x-coordinates?","$-6 - 2 = -8$."],["What is the difference in y-coordinates?","$-1 - 5 = -6$."],["Substitute the differences into the distance formula.","$\\sqrt{(-8)^2 + (-6)^2} = \\sqrt{64 + 36} = \\sqrt{100}$."],["What is the square root of 100?","$\\sqrt{100} = 10$"]],"answer":"10","solution":"We use the distance formula: $\\sqrt{(-6 - 2)^2 + (-1 - 5)^2},$ so then we find that $\\sqrt{64 + 36} = \\boxed{10}$.\n\n- OR -\n\nWe note that the points $(2, 5)$, $(-6, -1)$, and $(2, -1)$ form a right triangle with legs of length 6 and 8. This is a Pythagorean triple, so the length of the hypotenuse must be $\\boxed{10}$."},{"problem":"Find the $\\emph{positive}$ real number(s) $x$ such that $\\frac{1}{2}\\left( 3x^2-1\\right) = \\left( x^2-50x-10\\right)\\left( x^2+25x+5\\right)$.","steps":[["Can you apply substitutions for the equation so that it becomes a linear equation?", "Write $a = x^2-50x-10$ and $b = x^2+25x+5$."],["Substitute the equation with a and b.","the equation given becomes\n\\[\\frac{a+2b-1}{2} = ab,\\] so $0=2ab-a-2b+1=(a-1)(2b-1)$."],["What do we know about $x$ from $(a-1)(2b-1)=0$?","$a-1=x^2-50x-11=0$ or $2b-1=2x^2+50x+9=0$. The former has a positive root, $x=25 + 2\\sqrt{159}$, while the latter does not."]],"answer":"25 + 2\\sqrt{159}","solution":"Write $a = x^2-50x-10$ and $b = x^2+25x+5$.  Then the equation given becomes\n\\[\\frac{a+2b-1}{2} = ab,\\]so $0=2ab-a-2b+1=(a-1)(2b-1)$. Then $a-1=x^2-50x-11=0$ or $2b-1=2x^2+50x+9=0$. The former has a positive root, $x=\\boxed{25 + 2\\sqrt{159}}$, while the latter does not."},{"problem":"It took $4$ days for $75$ workers, all working together at the same rate, to build an embankment. If only $50$ workers had been available, how many total days would it have taken to build the embankment?","steps":[["How many total worker-days were required to build the embankment?","$75$ workers working for $4$ days would result in $75*4 = 300$ total worker-days."],["If only $50$ workers were available, how many days would it take to build the embankment?","If $75$ workers completed the project in $4$ days, then $50$ workers would need $300\/50 = 6$ days to complete the same project."]],"answer":"6","solution":"Since $\\text{work} = \\text{rate} \\times \\text{time}$, let $r$ be the rate at which one worker can built an embankment. It follows that 1 embankment takes \\[1\\text{ embankment}=(75r) \\times (4\\ \\text{days})\\] so $r = \\frac{1}{4 \\cdot 75}.$  If only $50$ workers were available, then  \\[1\\text{ embankment} = (50r) \\times (t\\ \\text{days})\\]  so \\[t = \\frac{1}{50 \\cdot \\frac{1}{4 \\cdot 75}} = \\frac{300}{50} = \\boxed{6}\\ \\text{days}.\\] Notice that the number of days and the number of workers are inversely related."},{"problem":"At a particular school with 43 students, each student takes chemistry, biology, or both. The chemistry class is three times as large as the biology class, and 5 students are taking both classes. How many people are in the chemistry class?","steps":[["Define variables for the number of students only in the biology class and only in the chemistry class.","Let $x$ be the number of students in the biology class who aren't in the chemistry class and $y$ be the number of students in the chemistry class who aren't in the biology class."],["Can we express the total number of students using x and y?","Since all students are in either one of the classes or in both, we know that $43=x+y+5$."],["Can you use the chemistry class being three times as large as the biology class to build another equation?","$3(x+5)=y+5$."],["Can you solve for $y$ in terms of $x$?","$y=3x+10$"],["Substitute $y=3x+10$ into the equation $43=x+(3x+10)+5$ to get $x$ and $y$.","$x=7$, and $y=31$"],["Add the number of chemistry students who aren't taking biology to the number of students taking both classes.","$31+5=36$."]],"answer":"36","solution":"Let $x$ be the number of students in the biology class who aren't in the chemistry class and $y$ be the number of students in the chemistry class who aren't in the biology class. Then, since all students are in either one of the classes or in both, we know that $43=x+y+5$. We also know that $3(x+5)=y+5$. Solving for $y$ in terms of $x$ gives us $y=3x+10$, and substituting that into the first equation gives us $43=x+(3x+10)+5$, which gives us $x=7$. Substituting this into the other equation gives us $y=31$. However, $y$ is only the number of chemistry students who aren't taking biology, so we need to add the number of students taking both to get our final answer of $\\boxed{36}$."}]

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
    if len(fewshot_examples) > 0:
        prompt = "You will be given an extremely difficult problem with an answer specified. Your job is to show the derivation to the answer. Specifically, your derivation should be structured as a list of sub-problems and solutions to the sub-problems, which will eventually lead to the specified answer.\n\n"
        for one in fewshot_examples:
            prompt += format_example(one, include_answer=True)
        prompt += format_example(test_example, include_answer=False)
    else:
        prompt = format_example(test_example, include_answer=False)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_split", type=str, default=None, help="Which split of the dataset to load.")
    parser.add_argument("--num_samples", type=int, default=0, help="a random number")
    parser.add_argument("--start", type=int, default=0, help="start of the dataset")
    parser.add_argument("--end", type=int, default=1000, help="end of the dataset")
    parser.add_argument("--port", type=str, default="30000", help="port number")
    parser.add_argument("--nofewshot", action="store_true", help="later iterations require no fewshot.")
    args = parser.parse_args()

    pro = start_server(args.model_name, args.port)
    sgl.set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))

    datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    test_examples = datasets[args.dataset_split].select(range(args.start, args.end))
    dataset_name = args.dataset_name.split('/')[-1]
    model_name = args.model_name.split('/')[-1]
    out_name = f"out/model-a-samples-{dataset_name}-{args.dataset_split}-{model_name}-num{args.num_samples}-start{args.start}-end{args.end}.json"
    if os.path.isfile(out_name):
        print("already exists:", out_name)
        outs = load_dataset("json", data_files=out_name, split="train").to_list()
        print(f"loaded first {len(outs)} examples")
        print(f"[{len(outs)}, {len(test_examples)}] examples remaining")
        test_examples = test_examples.select(range(len(outs), len(test_examples)))
    else:
        outs = []
    for one in tqdm(test_examples):
        if "mmlu" in args.dataset_name and '|' in one['question']:
            continue
        if "competition_math" in args.dataset_name:
            one['answer'] = extract_substrings(one['solution'])
        if 'problem' not in one:
            one['problem'] = deepcopy(one['question'])
            del one['question']
        if args.nofewshot:
            in_text = gen_prompt(one, [])
            gpt_answer = sample_completion(in_text, stop="Therefore, the answer is", samples=args.num_samples)
        else:
            in_text = gen_prompt(one, fewshot_examples)
            gpt_answer = sample_completion(in_text, stop="Problem:", samples=args.num_samples)
        one['output'] = gpt_answer
        outs.append(one)
 
        new_ds = Dataset.from_list(outs)
        new_ds.to_json(out_name)
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == '__main__':
    main()
