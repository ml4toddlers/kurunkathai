import os
import json
import datasets

data_dir = "chunks"
data = []

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(data_dir, filename)
        print(f"Loading {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            
            for item in json_data:
                for key, text in item.items(): 
                    data.append({"row": int(key), "text": text.replace("\u200b", "").replace("\n","")})
data.sort(key=lambda x: x["row"])
sorted_text_data = [{"text": item["text"]} for item in data]
dataset = datasets.DatasetDict({"train":datasets.Dataset.from_list(sorted_text_data)})
dataset.push_to_hub("tniranjan/tinystories_ta_google_translate")