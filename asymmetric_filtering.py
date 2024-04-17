import argparse
from copy import deepcopy
import re
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import sample_completion, load_model_and_tokenizer

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

def format_example(example, include_answer=True, chain=False):
    if include_answer:
        prompt = "Problem: " + example['problem'] + " The sub-problems are\n"
        for q, a in example['steps']:
            prompt += "Sub-problem: "+q+'\n'
        prompt += "START YOUR WORK BELOW:\n"
        for q, a in example['steps']:
            prompt += f"Sub-problem: {q}\nSolution to the sub-problem: {a}\n"
        prompt += "Answer: \\boxed{"+example['answer']+"}\n\n"
        return prompt
    if 'choices' in example:
        prompt = "Problem: " + example['problem'] + " Which of the following answer choices is correct?"
        choices = ["A", "B", "C", "D"]
        assert len(example['choices']) == len(choices)
        for j, item in enumerate(example['choices']):
            prompt += "\n{}. {}".format(choices[j], item)
        prompt += f"\nSolution: {choices[example['answer']]}. {example['choices'][example['answer']]}\n\n"
    else:
        all_prompts = []
        example['steps'] = []
        example['output'] = list(set(example['output']))
        for text in example['output']:
            # Mistral sometimes automatically generates next problem and we want to remove that
            text = text.split("Problem:")[0].strip()
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
                example['steps'].append(curr)
            else:
                continue
            prompt = "Problem: " + example['problem'] + '\n'
            for q, a in example['steps'][-1]:
                prompt += "Sub-problem: "+q+'\n'
            prompt += "START YOUR WORK BELOW:\n"
            all_prompts.append(prompt)
        del example['output']
    return all_prompts

def gen_prompt(test_example, fewshot_examples):
    prompt = "You will be given an extremely difficult problem and its decomposed sub-problems. Your job is to derive an answer by finding solutions to the sub-problems. Please highlight your solution with \\boxed{number} where number is the numerical answer without unit.\n\n"
    for one in fewshot_examples:
        prompt += format_example(one, include_answer=True)
    out = format_example(test_example, include_answer=False)
    if out is None:
        return None
    if isinstance(out, list):
        prompt = [prompt for _ in out]
        for i in range(len(prompt)):
            prompt[i] += out[i]
    else:
        prompt += out
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--num", type=int, default=0, help="a random number")
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer(args.model_name)
    test_examples = load_dataset("json", data_files=args.dataset_name, split="train")
    out_name = args.dataset_name.replace(".json", "-asymmetric-filtered.json")
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
        in_text = gen_prompt(one, fewshot_examples)
        gpt_answer = sample_completion(in_text, model, tok, samples=1)
        one['model-b'] = gpt_answer
        outs.append(one)
    
        new_ds = Dataset.from_list(outs)
        new_ds.to_json(out_name)


if __name__ == '__main__':
    main()
