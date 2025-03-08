import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
from dotenv import load_dotenv
import tqdm
import json
load_dotenv()

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("API key not found in .env file.")

gpt3_5 = "gpt-3.5-turbo"
gpt4o = "gpt-4o-2024-08-06"
gpt4o_mini="gpt-4o-mini"


client = openai.OpenAI(api_key=OPENAI_API_KEY)

def rate_stories(stories):
    formatted_stories = "\n\n".join([f"Story {i+1}: {story}" for i, story in enumerate(stories)])

    system_message = """You are a strict story evaluator. 
    Always return exactly four numbers for each story in this order: Creativity,Consistency,Grammar,Plot.
    Format example: 7,8,9,6
    Do not add any extra text or explanations."""

    user_prompt = f"""
    Rate each Tamil story below on a scale of 0-10 for Creativity, Consistency, Grammar, and Plot.

    {formatted_stories}

    Respond with one line per story in the format: Creativity,Consistency,Grammar,Plot.
    """

    response = client.chat.completions.create(
        model=gpt4o_mini,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": user_prompt}],
        max_tokens=10 * len(stories),  
        temperature=0
    )

    scores_list = response.choices[0].message.content.strip().split("\n")
    ratings = []

    for i, scores in enumerate(scores_list):
        scores = scores.strip()  # Remove spaces
        if scores and "," in scores:  # Ensure valid format
            score_values = scores.split(",")
            while len(score_values) < 4:
                score_values.append("0")  

            try:
                creativity, consistency, grammar, plot = map(int, score_values[:4])  # Ensure only 4 values
                ratings.append({
                    "Creativity": creativity,
                    "Consistency": consistency,
                    "Grammar": grammar,
                    "Plot": plot,
                    "Story": stories[i]
                })
            except ValueError:
                print(f"Skipping invalid response for Story {i+1}: {scores}")  # Debugging output
        else:
            print(f"Skipping empty/invalid line for Story {i+1}: {scores}")  # Debugging output

    return ratings


def generate_and_rate(model_config):
    with open(model_config, "r") as f:
        model_config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
    model = AutoModelForCausalLM.from_pretrained(**model_config["model_path_dict"])
    test_prompts = json.load(open("evaluation/test.json"))
    test_prompts = [prompt for group in test_prompts.values() for prompt in group]
    generated_stories = []

    for prompt in tqdm.tqdm(test_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens= 250, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=100, attention_mask=input_ids.ne(tokenizer.pad_token_id))
        generated_stories.append(tokenizer.decode(output[0], skip_special_tokens=True))

    ratings = rate_stories(generated_stories)
    json.dump(ratings, open(model_config.replace("model_config", "eval"), "w"), indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    args = parser.parse_args()
    generate_and_rate(args.model_config)