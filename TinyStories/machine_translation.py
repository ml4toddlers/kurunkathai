from deep_translator import GoogleTranslator
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def translate_bulk(inputs):
    translator = GoogleTranslator(source='en', target='ta')
    translations = translator.translate_batch(inputs)
    outputs = [{"text":translation} for translation in translations]
    return outputs

def collate_text(batch):
    return [item["text"] for item in batch]

def save_dataset(dataset_dict, path):
    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk(path)

def main():
    dataset = load_dataset("roneneldan/TinyStories")
    ta_dataset = {}
    for key, split in dataset.items():
        ta_dataset[key] = []
        dataloader = DataLoader(split, batch_size = 500, shuffle = False, collate_fn = collate_text)
        for index,batch in  tqdm(enumerate(dataloader), "Processing data", total=len(dataloader)):
            translated_batch = translate_bulk(batch)
            print(len(batch), len(translated_batch))
            ta_dataset[key].extend(translated_batch)
            if index%20 == 1:
                save_dataset(ta_dataset, "ta_tinystories_google_translate")
    save_dataset(ta_dataset, "ta_tinystories_google_translate")

if __name__ == "__main__":
    main()


