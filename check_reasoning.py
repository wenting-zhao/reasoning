import argparse
from copy import deepcopy
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import query_chat, load_model_and_tokenizer

fewshot_examples =[{"problem":"If $1+2x+3x^2 + \\dotsb=9$, find $x$.","level":"Level 3","type":"Intermediate Algebra","solution":"Let $S = 1 + 2x + 3x^2 + \\dotsb.$  Then\n\\[xS = x + 2x^2 + 3x^3 + \\dotsb.\\]Subtracting these equations, we get\n\\[(1 - x) S = 1 + x + x^2 + \\dotsb = \\frac{1}{1 - x},\\]so $S = \\frac{1}{(1 - x)^2}.$  Thus, we want to solve\n\\[\\frac{1}{(1 - x)^2}  = 9.\\]then $(1 - x)^2 = \\frac{1}{9},$ so $1 - x = \\pm \\frac{1}{3}.$  Since $x$ must be less than 1, $1 - x = \\frac{1}{3},$ so $x = \\boxed{\\frac{2}{3}}.$","answer":"\\frac{2}{3}","output":"Sub-problem: What is the sum of an infinite geometric series when $|r|<1$?\nSolution to the sub-problem: The sum of an infinite geometric series is $\\frac{a}{1-r}$, where $a$ is the first term and $r$ is the common ratio.\nSub-problem: Can we rewrite the given series as an infinite geometric series?\nSolution to the sub-problem: The given series $1+2x+3x^2+\\dotsb$ can be rewritten as $1+2x+3x^2+\\dotsb=\\sum_{n=0}^{\\infty} (n+1)x^n$.\nSub-problem: What is the first term and common ratio of the rewritten series?\nSolution to the sub-problem: The first term is $a=1$ and the common ratio is $r=x$.\nSub-problem: Substitute $a=1$ and $r=x$ into the sum formula and set it equal to 9.\nSolution to the sub-problem: $\\frac{1}{1-x}=9$.\nSub-problem: Solve the equation $\\frac{1}{1-x}=9$ to find the value of $x$.\nSolution to the sub-problem: $\\frac{1}{9} = 1-x$, so $x = 1 - \\frac{1}{9} = \\frac{8}{9} = \\frac{2}{3}$.","judge":"Because \\frac{8}{9} is not equal to \\frac{2}{3}$, the solution is wrong."},{"problem":"Sam is hired for a 20-day period. On days that he works, he earns $\\$$60. For each day that he does not work, $\\$$30 is subtracted from his earnings. At the end of the 20-day period, he received $\\$$660. How many days did he not work?","level":"Level 3","type":"Algebra","solution":"Call $x$ the number of days Sam works and $y$ the number of days he does not. We can set up the following system of equations to represent the given information: \\begin{align*}\nx+y &= 20 \\\\\n60x - 30y &= 660 \\\\\n\\end{align*} The first equation represents the total number of days Sam works, and the second equation represents his total profit. Solving for $x$ in the first equation yields $x = 20 - y$. Substituting into the second equation gives $60(20-y) - 30y = 660$. Canceling a factor of $10$ and multiplying out gives $120 - 6y - 3y = 66$. This simplifies to $-9y = -54$, or $y = 6$. Thus, Sam did not work for $\\boxed{6}$ days.","answer":"6","output":"Sub-problem: Let $x$ be the number of days Sam worked and $y$ be the number of days he did not work.\nSolution to the sub-problem: The total days in the period are $x + y = 20$.\nSub-problem: Can you express Sam's earnings depending on the number of days he worked and did not work?\nSolution to the sub-problem: Sam earns $60x - 30y$.\nSub-problem: Can you create an equation with Sam's total earnings over the 20-day period?\nSolution to the sub-problem: His total earnings are $660$, so we have $60x - 30y = 660$.\nSub-problem: Can you use $x + y = 20$ to substitute for $x$ in the earnings equation?\nSolution to the sub-problem: Substituting $x = 20 - y$ into $60x - 30y = 660$ gives $60(20 - y) - 30y = 660$.\nSub-problem: Solve the equation $60(20 - y) - 30y = 660$.\nSolution to the sub-problem: $1200 - 60y - 30y = 660$, so $90 - 90y = 660$, which simplifies to $-90y = -540$ and $y = 6$. \n\nTherefore, Sam did not work for 6 days.","judge":"The solution is correct."},{"problem":"Find the value of $x$ that satisfies $\\frac{\\sqrt{3x+5}}{\\sqrt{6x+5}}=\\frac{\\sqrt{5}}{3}$. Express your answer as a common fraction.","level":"Level 4","type":"Algebra","solution":"We begin by cross multiplying and then squaring both sides \\begin{align*}\n\\frac{\\sqrt{3x+5}}{\\sqrt{6x+5}}&=\\frac{\\sqrt{5}}{3}\\\\\n3\\sqrt{3x+5}&=\\sqrt{5}\\cdot\\sqrt{6x+5}\\\\\n\\left(3\\sqrt{3x+5}\\right)^2&=\\left(\\sqrt{5}\\cdot\\sqrt{6x+5}\\right)^2\\\\\n9(3x+5) &=5(6x+5)\\\\\n20 &= 3x\\\\\nx&=\\boxed{\\frac{20}{3}}.\\\\\n\\end{align*}Checking, we see that this value of $x$ satisfies the original equation, so it is not an extraneous solution.","answer":"\\frac{20}{3}","output":"Sub-problem: Square both sides to get rid of the square roots.\nSolution to the sub-problem: $\\frac{3x+5}{6x+5}=\\frac{5}{9}$.\nSub-problem: Cross multiply to solve for $x$.\nSolution to the sub-problem: $27x+45=30x+25$, $5x=20$, $x=\\frac{20}{5}=\\frac{20}{3}$, which is the final answer.","judge":"Because $27x+45=30x+25$ does not become $5x=20$, the solution is wrong."},{"problem":"If $f(3)=1$ and $f(2x)=2f(x)$ for all $x$, find $f^{-1}(64)$.","level":"Level 5","type":"Algebra","solution":"We are looking for some $x$ such that $f(x)=64$.  We notice that by doubling $x$ we can double $f(x)$ as well and also that $f(3)=1$.\n\nApplying $f(2x)=2f(x)$ repeatedly, we have: \\begin{align*}\nf(3)&=1,\\\\\nf(6)&=2,\\\\\nf(12)&=4,\\\\\nf(24)&=8,\\\\\nf(48)&=16,\\\\\nf(96)&=32,\\\\\nf(192)&=64.\n\\end{align*}So $f^{-1}(64)=\\boxed{192}$.","answer":"192","output":"Sub-problem: Can we find $f(6)$ using $f(2x) = 2f(x)$?\nSolution to the sub-problem: Since $f(2*3) = 2*f(3)$, we have $f(6) = 2*1 = 2$.\nSub-problem: Use the fact that $f^{-1}(y) = x$ if $f(x) = y$ to find $f^{-1}(64)$.\nSolution to the sub-problem: We know that $f(192) = 64$ because $f(192) = 2*f(96) = 2*32 = 64$. Therefore, $f^{-1}(64) = 192$.","judge":"The solution is correct."}]

def format_example(example, include_answer=True):
    prompt = "Problem: " + example['problem'] + f"\nSolution: {example['output']}\n"
    if include_answer:
        prompt += example['judge']
    prompt += "\n\n"
    return prompt

def gen_prompt(test_example, fewshot_examples):
    prompt = "You are a teacher. Your task is to judge whether the solution to the problem is correct or wrong.\n\n"
    for one in fewshot_examples:
        prompt += format_example(one)
    prompt += format_example(test_example, include_answer=False)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer(args.model_name)
    test_examples = load_dataset('json', data_files=args.dataset_name, split="train")
    outs = []
    for one in tqdm(test_examples):
        in_text = gen_prompt(one, fewshot_examples)
        gpt_answer = query_chat([{'role': 'user', 'content': in_text}], model=model)
        one['judge'] = gpt_answer
        outs.append(one)
    
    test_examples = Dataset.from_list(outs)
    test_examples.to_json(args.dataset_name.replace('.json', '_judged.json'))

if __name__ == '__main__':
    main()
