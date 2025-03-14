from ast import mod
from genericpath import isfile
import re
from turtle import st
import generate
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
from dotenv import load_dotenv
import tqdm
import torch
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

def rate_story(stories, language="Tamil"):
    backbones = [gpt3_5, gpt4o]
    all_ratings = {backbone: {"ratings":[], "avg_ratings":{"Creativity": 0, "Consistency": 0, "Grammar": 0, "Plot": 0}} for backbone in backbones}        
    for story in stories:
        system_message = """You are a strict story evaluator. 
        Always return exactly four numbers for the following story in this order: Creativity,Consistency,Grammar,Plot.
        Format example: 7,8,9,6
        Do not add any extra text or explanations."""

        user_prompt = f"""
        Rate the {language} story below on a scale of 0-10 for Creativity, Consistency, Grammar, and Plot.

        {story}

        Respond with one line per story in the format: Creativity,Consistency,Grammar,Plot.
        """
        for backbone in backbones:
            response = client.chat.completions.create(
                model=backbone,
                messages=[{"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}],
                max_tokens=10 ,  
                temperature=0
            )

            scores_list = response.choices[0].message.content.strip().split("\n")

            for i, scores in enumerate(scores_list):
                scores = scores.strip()  # Remove spaces
                if scores and "," in scores:  # Ensure valid format
                    score_values = scores.split(",")
                    while len(score_values) < 4:
                        score_values.append("0")  

                    try:
                        creativity, consistency, grammar, plot = map(int, score_values[:4])  # Ensure only 4 values
                        all_ratings[backbone]["avg_ratings"]["Creativity"] += creativity
                        all_ratings[backbone]["avg_ratings"]["Consistency"] += consistency
                        all_ratings[backbone]["avg_ratings"]["Grammar"] += grammar
                        all_ratings[backbone]["avg_ratings"]["Plot"] += plot
                        all_ratings[backbone]["ratings"].append({
                            "Creativity": creativity,
                            "Consistency": consistency,
                            "Grammar": grammar,
                            "Plot": plot,
                            "Story": story
                        })
                    except ValueError:
                        print(f"Skipping invalid response for Story {i+1}: {scores}")  # Debugging output
                else:
                    print(f"Skipping empty/invalid line for Story {i+1}: {scores}")  # Debugging output
    for backbone in all_ratings:
        for key in all_ratings[backbone]["avg_ratings"]:
            all_ratings[backbone]["avg_ratings"][key] /= len(stories)
    
    return all_ratings


def generate_and_rate(model_config_path, backbone="gpt4o"):
    if os.path.isfile(model_config_path):
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config = json.loads(model_config_path)
        path_dict = model_config["model_path_dict"]
        model_config_path = "evaluation/model_config_"+path_dict["pretrained_model_name_or_path"].replace("tniranjan/","") +path_dict["revision"][:6]+".json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
    model = AutoModelForCausalLM.from_pretrained(**model_config["model_path_dict"]).to(device)
    file_paths = ["evaluation/test.json", "evaluation/validation.json"]
    avg_ratings = {}
    for file_path in file_paths:
        label = file_path.split("/")[-1].split(".")[0]
        avg_ratings[label] = {}
        test_prompts = json.load(open(file_path, "r",encoding="utf-8"))
        test_prompts = [prompt for group in test_prompts.values() for prompt in group]
        generated_stories = []

        for prompt in tqdm.tqdm(test_prompts, desc=f"Generating for {label} with {model_config_path.split('/')[-1].replace('.json','').replace('model_config_','')}"):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens= 250, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=80, attention_mask=input_ids.ne(tokenizer.pad_token_id))
            story =tokenizer.decode(output[0], skip_special_tokens=True)
            generated_stories.append(f"""{story}""")
        json.dump(generated_stories, open(model_config_path.replace("model_config", "stories_" + label), "w",encoding="utf-8"), ensure_ascii=False, indent=4)
        # generated_stories = json.load(open(model_config_path.replace("model_config", "stories_" + label), "r",encoding="utf-8"))
        all_ratings = rate_story(generated_stories)
        for backbone in all_ratings:
            print(f"Average ratings for {label} set with {backbone}:" + str(all_ratings[backbone]["avg_ratings"]))
            avg_ratings[label][backbone] = all_ratings[backbone]["avg_ratings"]
        json.dump(all_ratings, open(model_config_path.replace("model_config", "eval_" + label), "w",encoding="utf-8"), ensure_ascii=False, indent=4)
    return avg_ratings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    args = parser.parse_args()
    generate_and_rate(args.model_config)