import huggingface_hub
import transformers
from datasets import load_dataset
import math
import torch
import tqdm
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
no_instruct_dataset = load_dataset("tniranjan/aitamilnadu_tamil_stories_no_instruct")
tokenizer = transformers.AutoTokenizer.from_pretrained("abhinand/tamil-llama-7b-instruct-v0.2")
all_checkpoints = huggingface_hub.HfApi().list_repo_commits("tniranjan/gptNeo-ta")[:-1]
text_hq = " ".join(no_instruct_dataset["validation"]["text"])
tokens_hq = tokenizer(text_hq, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
batch_size=1024
stride = batch_size // 2
 
def compute_perplexity(model, tokens, idx):
    nll_loss = 0.0
    total_tokens = 0
    for i in tqdm.tqdm(range(0, tokens.size(0), stride),desc=f"Checkpoint:{idx} Perplexity"):
        input_ids = tokens[i : i + batch_size].unsqueeze(0).to(device)
        if input_ids.size(1) < 2:  
            continue
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss  
        n_tokens = input_ids.size(1) - 1
        nll_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = nll_loss / total_tokens
    return math.exp(avg_loss)


perplexities=[]
for idx, checkpoint in enumerate(tqdm.tqdm(all_checkpoints, desc=f"Checkpoint")):
    model = transformers.GPTNeoForCausalLM.from_pretrained("tniranjan/gptNeo-ta", revision = checkpoint.commit_id)
    model.eval()
    model.to(device)
    perplexity = compute_perplexity(model, tokens_hq, idx)
    perplexities.append({"commit_id":checkpoint.commit_id,"iter":int(checkpoint.title.split("step ")[-1]), "perplexity":perplexity})

    print(f"Checkpoint {idx} :Perplexity: {perplexity}")
json.dump( perplexities, open("perplexity.json","w"),indent=4)
