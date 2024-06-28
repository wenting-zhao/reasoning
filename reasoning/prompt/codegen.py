import datasets
import json
import re

from reasoning.prompt.prompt import Prompt


NEWLINE = "\n\n"


class AppsStdInPrompt(Prompt):
    ZEROSHOT_STRING = """You are an expert programmer.
You will be given a problem to solve. Please write code for a solution.

The problem will require you to read input from stdin, and print the solution to stdout.

# Solution
```python
# code solution here
```"""


    def format_fewshot_example(self, example):
        return f"""# Example Problem
{example["question"].strip()}

# Example Solution
```python
{json.loads(example["solutions"])[0].strip()}
```"""

    def get_fewshot_examples(self, k):
        train = datasets.load_dataset("codeparrot/apps", split="train")
        return [
            self.format_fewshot_example(x) for x in train.select(range(4))
        ]

    def render(self, example):
        # sglang format
        return [
            {
                "role": "system",
                "content": self.ZEROSHOT_STRING
                if self.use_fewshot
                else f"{self.ZEROSHOT_STRING}\n{NEWLINE.join(self.fewshot_examples)}",
            },
            {
                "role": "user",
                "content": f"This is the problem:\n\n# Problem\n{example['question']}\n# Solution",
            },
        ]


class AppsCallPrompt(Prompt):
    ZEROSHOT_STRING = """You are an expert programmer.
You will be given a problem to solve. Please write code for a solution.

The problem will require you to complete the following starter code.
You must copy the starter code then finish it.

# Starter code
```python
# starter code here
```

# Solution
```python
# copy and complete starter code here
```"""


    def format_fewshot_example(self, example):
        return f"""# Example Problem
{example["question"].strip()}

# Example Starter Code
```python
{example["starter_code"].strip()}
```

# Example Solution
```python
{json.loads(example["solutions"])[0].strip()}
```"""

    def get_fewshot_examples(self, k):
        train = (
            datasets.load_dataset("codeparrot/apps", split="train")
            .filter(lambda x: x["starter_code"] != "")
        )
        return [
            self.format_fewshot_example(x) for x in train.select(range(k))
        ]

    def render(self, example):
        # sglang format
        return [
            {
                "role": "system",
                "content": self.ZEROSHOT_STRING
                if not self.use_fewshot
                else f"{self.ZEROSHOT_STRING}\n{NEWLINE.join(self.fewshot_examples)}",
            },
            {
                "role": "user",
                "content": f"This is the problem:\n\n# Problem\n{example['question']}\n\n# Starter Code\n```python\n{example['starter_code'].strip()}\n```\n\n# Solution",
            },
        ]


class AppsStdInPlanPrompt(Prompt):
    ZEROSHOT_STRING = """You are an expert programmer.
You will be given a problem to solve.

First, list out the steps and helper functions needed to solve the task in the following format:
# Plan
1. function1: Type -> Type -> Type. Description.
2. function2: Type -> Type -> Type. Description.

# Solution
```python
# code solution here
```"""

    def format_fewshot_example(example, attempt):
        return f"""# Example Problem
{example["question"]}

# Plan
{example["plan"][attempt]}

# Solution
{example["code"][attempt]}"""


    def render(example):
        raise NotImplementedError
        return [
            {
                "role": "system",
                "content": ZEROSHOT_STRING
                if self.use_fewshot
                else f"{ZEROSHOT_STRING}\n{NEWLINE.join(self.fewshot_examples)}",
            },
            {
                "role": "user",
                "content": f"This is the problem:\n\n# Problem\n{example['question']}\n# Solution",
            },
        ]


def parse_response(text):
    # plan_re = r'```plan(.*)```'
    # plan_re = r'# Plan\n(.*)# Solution'
    # python_re = r'```python(.*)```'
    python_re = r"```(?:python)?(.*)```"
    # plan_matches = re.findall(plan_re, text, re.DOTALL)
    python_matches = re.findall(python_re, text, re.DOTALL)

    # generation failure => parsing failure
    # if len(plan_matches) == 0:
    # plan_matches = [""]
    if len(python_matches) == 0:
        python_matches = [""]

    # return python_matches[0], plan_matches[0]
    return python_matches[0], None


if __name__ == "__main__":
    stdin_prompt = AppsStdInPrompt(use_fewshot=True, k=4)
    call_prompt = AppsCallPrompt(use_fewshot=True, k=4)
    example = {
        "question": "blah",
        "starter_code": "class Blah:",
    }
    stdin_prompt = stdin_prompt.render(example)
    call_prompt = call_prompt.render(example)

    print("STDIN")
    for message in stdin_prompt:
        print(message["content"])
    print("CALL")
    for message in call_prompt:
        print(message["content"])
    import pdb; pdb.set_trace()
