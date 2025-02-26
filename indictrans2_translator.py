import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Source and target languages
src_lang, tgt_lang = "eng_Latn", "tam_Taml"

# Model name from Hugging Face
model_name = "ai4bharat/indictrans2-en-indic-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16,  # Performance boost with float16
    attn_implementation="flash_attention_2"
).to(DEVICE)

# Initialize IndicTrans processor
ip = IndicProcessor(inference=True)

# Function to translate text to Tamil
def translate_to_tamil(input_sentences, model, tokenizer, ip):
    batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

    # Tokenize the batch
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate translations
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode translated text
    with tokenizer.as_target_tokenizer():
        translations = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess translations
    return ip.postprocess_batch(translations, lang=tgt_lang)

# Load dataset (both train & test splits)
dataset = load_dataset("roneneldan/TinyStories")

# Define batch size
batch_size = 40  # Adjust based on available memory

# Function to process and save each split
def process_split(split_name, data):
    data_list = data.to_list()  # Convert to list for easy manipulation
    pbar = tqdm(total=len(data_list), desc=f"Translating {split_name} to Tamil")

    # Process in batches
    for i in range(0, len(data_list), batch_size):
        batch_texts = [entry["text"] for entry in data_list[i : i + batch_size]]

        # Translate the batch
        translated_texts = translate_to_tamil(batch_texts, model, tokenizer, ip)

        # Store translations in dataset
        for j, translation in enumerate(translated_texts):
            data_list[i + j]["text_tamil"] = translation

        # Save progress every 10,000 entries
        if (i + batch_size) % 10000 == 0:
            dataset[split_name] = Dataset.from_list(data_list)
            dataset.save_to_disk("TinyStories_Tamil")
            dataset.push_to_hub("tniranjan/TinyStories_Tamil")

        # Update progress bar
        pbar.update(batch_size)

    # Save final version of split
    dataset[split_name] = Dataset.from_list(data_list)
    dataset.save_to_disk("TinyStories_Tamil")
    dataset.push_to_hub("tniranjan/TinyStories_Tamil")

    pbar.close()
    print(f"Translation completed for {split_name} split.")

# Process both train and test splits
for split in ["train", "test"]:
    process_split(split, dataset[split])

print("All translations completed and dataset saved.")
