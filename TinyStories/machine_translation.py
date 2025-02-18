import asyncio
from googletrans import Translator
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

async def translate_bulk(inputs):
    async with Translator() as translator:
        translations = await translator.translate(inputs, dest='ta')

    if not isinstance(translations, list):
        translations = [translations]
    
    return [{"text":translation.text for translation in translations}]

def collate_text(batch):
    return [item["text"] for item in batch]

async def main():
    dataset = load_dataset("roneneldan/TinyStories")
    ta_dataset = {key:[] for key in dataset.keys()}
    for key, split in dataset.items():
        dataloader = DataLoader(split, batch_size = 50, shuffle = False, collate_fn = collate_text)
        for batch in  tqdm(dataloader, "Processing data", total=len(dataloader)):
            translated_batch = await translate_bulk(batch)
            ta_dataset[key].append(translated_batch)
            
    ta_tinystories = Dataset.from_dict(ta_dataset)
    ta_tinystories.save_to_disk("ta_tinystories_google_translate")

if __name__ == "__main__":
    asyncio.run(main())


