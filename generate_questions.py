import argparse
from copy import deepcopy
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import query_chat, load_model_and_tokenizer

fewshot_examples=[{"problem":"Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?","steps":[["How much does Tina make in an 8-hour shift?","She works 8 hours a day for $18 per hour so she makes 8*18 = 144.00 per 8-hour shift"],["How many hours of overtime does Tina get?","She works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = 2 hours of overtime"],["How much does Tina make in overtime?","Overtime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*.5 = 9.00"],["How much does Tina make in overtime?","Her overtime pay is 18+9 = 27.00"],["How much does Tina make in a week?","Her base pay is $144.00 per 8-hour shift and she works 5 days and makes 5 * $144 = 720.00"],["How much does Tina make in overtime?","Her overtime pay is $27.00 per hour and she works 2 hours of overtime per day and makes 27*2 = 54.00 in overtime pay"],["How much does Tina make in overtime?","2 hours of overtime pay for 5 days means she makes 54*5 = $270.00"],["How much does Tina make in a week?","In 5 days her base pay is $720.00 and she makes $270.00 in overtime pay so she makes $720 + $270 = 990.00"]],"answer":"990"},{"problem":"Gail has two fish tanks. The first tank is twice the size of the second tank. There are 48 gallons of water in the first tank. She follows the rule of one gallon of water per inch of fish. If she keeps two-inch fish in the second tank and three-inch fish in the first tank, how many more fish would Gail have in the first tank than the second tank if one of the first tank fish eats another?","steps":[["How many gallons are in the second tank?","The second tank is 48 / 2 = 24 gallons."],["How many two-inch fish does Gail keep in the second tank?","Following her rule, Gail keeps 24 / 2 = 12 two-inch fish in the second tank."],["How many fish does Gail keep in the first tank?","She keeps 48 / 3 = 16 fish in the first tank."],["How many fish would Gail have in the first tank if one of the first tank fish eats another?","If one fish in the first tank ate another, she would have 16 - 1 = 15 fish in the first tank."],["How many more fish would Gail have in the first tank than the second tank if one of the first tank fish eats another?","Thus, Gail would have 15 - 12 = 3 more fish in the first tank."]],"answer":"3"},{"problem":"Joseph and his friends watched two movies in his house. The first movie is 1 hour and 30 minutes long while the second movie is 30 minutes longer than the first. Before the movies, they spent 10 minutes making popcorn and twice as long making fries. How long, in hours, did it take Joseph and his friends to cook and watch the movies?","steps":[["How long was the first movie?","The first movie was 60 + 30 = 90 minutes long since an hour has 60 minutes."],["How long was the second movie?","The second movie was 90 + 30 = 120 minutes long."],["How long did it take them to watch the two movies?","It took them a total of 90 + 120 = 210 minutes to watch the two movies."],["How long did it take them to cook the fries?","It took them 10 x 2 = 20 minutes to cook the fries."],["How long did it take them to cook?","Thus, it took them a total of 10 + 20 = 30 minutes to cook."],["How long did it take Joseph and his friends to cook and watch the movies?","So, they spent 210 + 30 = 240 minutes watching the movies and cooking."],["How long, in hours, did it take Joseph and his friends to cook and watch the movies?","In hours, this is equal to 240/60 = 4 hours."]],"answer":"4"},{"problem":"An elementary school teacher is making Halloween goodie bags for her class. She wants the bags to be personalized, so she surveys her students asking whether they'd like a vampire-themed bag or a pumpkin-themed bag. Of her 25 students, 11 indicate they want the vampire-themed bag and 14 indicate they want the pumpkin-themed bag. The store the teacher shops at sells packs of 5 of each theme at a price of $3 per package, as well as individual bags of each theme at a price of $1 each. What is the least amount of money the teacher can spend on the bags if she buys every student the theme they requested?","steps":[["How much will the teacher spend on the vampire theme?","Because it is cheaper to buy the packs of 5, the teacher should satisfy the 11 students who want the vampire theme with 2 packs of 5 and 1 individual bag. This will cost the teacher 2*3 + 1*1 = 7."],["How much will the teacher spend on the pumpkin theme?","Similarly, the 14 students who want a pumpkin-themed bag can be satisfied by 2 packs of 5 and 4 individual bags at a cost of 2*3 + 4*1 = 10."],["How much will the teacher spend on the bags?","Therefore, the teacher must spend 7 + 10 = 17."]],"answer":"17"}]


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
    prompt = "Problem: " + example['problem'] + f"\nAnswer: {example['answer']}\n\n"
    if include_answer:
        for q, a in example['steps']:
            prompt += f"Sub-question: {q}\nAnswer: {a}\n"
        prompt += "\n"
    return prompt

def gen_prompt(test_example, fewshot_examples):
    prompt = "You will be given an extremely difficult problem with an answer specified. Your job is to show the derivation to the answer. Specifically, your derivation should be structured as a list of sub-questions and answers to the sub-questions, which will eventually lead to the specified answer.\n\n"
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
