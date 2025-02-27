import requests
import json
import os
from datasets import load_dataset, Dataset, DatasetDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
import logging

# API Endpoint
API_URL = "https://admin.models.ai4bharat.org/inference/translate"

# Output File
OUTPUT_FILE = "tinystories_translated.json"
BATCH_SIZE = 1000  # Save every 1000 translations
MAX_WORKERS = 10   # Number of parallel threads

# Setup logging
logger = logging.getLogger("translator")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("translation.log", maxBytes=10*1024*1024, backupCount=5)
logger.addHandler(handler)

# Load the TinyStories dataset (train split)
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Resume if partially completed
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        translated_data = json.load(f)
else:
    translated_data = [{"text": None} for _ in range(len(dataset))]  # Placeholder for all entries

# Function to call translation API
def translate_text(index, text):
    payload = {
        "sourceLanguage": "en",
        "targetLanguage": "ta",  # Translate to Tamil
        "input": text,
        "task": "translation",
        "serviceId": "ai4bharat/indictrans--gpu-t4",
        "track": True
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        result = response.json()
        return index, result.get("output", "Translation Failed")
    except requests.exceptions.RequestException as e:
        logger.error(f"Translation error at index {index}: {e}")
        return index, "Translation Error"

# Process dataset in parallel
def process_dataset():
    global translated_data
    original_texts = dataset["text"]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(translate_text, i, text): i for i, text in enumerate(original_texts)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating Stories"):
            index, translated_text = future.result()
            translated_data[index] = {"text": translated_text}  # Maintain original order

            # Save checkpoint every BATCH_SIZE entries
            if index % BATCH_SIZE == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(translated_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Checkpoint saved at {index} translations")

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    logger.info("Final save completed.")

# Convert translated data to Hugging Face dataset
def save_as_huggingface_dataset():
    translated_dataset = Dataset.from_dict({"text": [entry["text"] for entry in translated_data]})
    dataset_dict = DatasetDict({"train": translated_dataset})

    dataset_dict.save_to_disk("tinystories_translated_hf")
    logger.info("Saved as Hugging Face dataset.")

# Run the script
if __name__ == "__main__":
    process_dataset()
    save_as_huggingface_dataset()
    print("Translation completed. Hugging Face dataset saved at tinystories_translated_hf")
