from deep_translator import GoogleTranslator
from datasets import load_dataset
import argparse
import json
import os
from tqdm import tqdm

def translate_indices(ds, split, start_index, num_batches, batch_size=500, odir="chunks"):
    end_index = start_index + num_batches * batch_size
    translator = GoogleTranslator(source='en', target='ta')
    translated_sublist = []
    for i in tqdm(range(start_index, end_index, batch_size),desc=f"Translating {split}_{start_index}_{end_index}"):
        batch =  ds[split][i:i+batch_size]["text"]
        translated = translator.translate_batch(batch=batch)
        translated_sublist.extend([ {idx+start_index:item} for idx,item in enumerate(translated)])
    with open(f"{odir}/{split}_{start_index}_{end_index}.json", "w", encoding='utf-8') as f:
        json.dump(translated_sublist, f, ensure_ascii=False, indent=4)
    return translated_sublist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--num_batches", type=int)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--odir", type=str, default="chunks")
    args = parser.parse_args().__dict__
    args["ds"] = load_dataset(args["ds"])
    os.makedirs(args["odir"], exist_ok=True)
    translated_sublist = translate_indices(**args)

if __name__ == "__main__":
    main()