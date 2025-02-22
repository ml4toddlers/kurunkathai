from transformers import GPTNeoConfig, GPTNeoForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import torch
import math
import argparse
import os

def train_val_split(dataset, split_ratio=0.9):
    dataset = dataset["train"].train_test_split(test_size=1-split_ratio, seed=42)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = logits.mean()  
    perplexity = math.exp(loss)  
    return {"loss": loss, "perplexity": perplexity}

def train(output_dir, log_dir):
        
    tokenizer = AutoTokenizer.from_pretrained("abhinand/tamil-llama-7b-instruct-v0.2")


    # dataset_high_quality =  DatasetDict({"validation":load_dataset("aitamilnadu/tamil_stories")["train"]})
    
    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=1024)
      
    if os.path.exists(output_dir+"/tokenized_tamil_CulturaX_dataset"):
        tokenized_dataset = DatasetDict.load_from_disk(output_dir+"/tokenized_tamil_CulturaX_dataset")
    else:
        dataset = load_dataset("uonlp/CulturaX", "ta")
        dataset = train_val_split(dataset)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.save_to_disk(output_dir+"/tokenized_tamil_CulturaX_dataset")
    # tokenized_hq_dataset = dataset_high_quality.map(tokenize_function, batched=True)

    tokenizer.save_pretrained(output_dir)
    config = GPTNeoConfig(
        vocab_size=tokenizer.vocab_size,
        attention_types=[[['global', 'local'], 4]],
        num_layers=8,
        hidden_size=1024,
        max_position_embeddings=1024,
    )

    model = GPTNeoForCausalLM(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="steps",
        save_steps = 10000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate = 1e-4,
        max_grad_norm=1.0,
        num_train_epochs=2,
        save_total_limit=4,
        logging_dir=log_dir,
        logging_steps=500,
        do_train=True,
        do_eval=True,
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=8,
        fp16=False,
        push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/gptNeo-ta/")
    parser.add_argument("--log_dir", type=str, default="logs/gptNeo-ta/")
    args = parser.parse_args()
    train(args.output_dir, args.log_dir)
