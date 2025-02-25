import os
import math
import argparse
import torch
import pytorch_lightning as pl
from transformers import (GPTNeoConfig, GPTNeoForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, AdamW, get_linear_schedule_with_warmup)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

class GPTNeoLightning(pl.LightningModule):
    def __init__(self, config, tokenizer, train_dataset, val_dataset, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.model = GPTNeoForCausalLM(config)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        perplexity = torch.exp(loss)
        self.log("val_perplexity", perplexity, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=1000, num_training_steps=len(self.train_dataloader()) * self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8, shuffle=True, collate_fn=self.data_collator)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=False, collate_fn=self.data_collator)

def train(output_dir, log_dir):
    tokenizer = AutoTokenizer.from_pretrained("abhinand/tamil-llama-7b-instruct-v0.2")

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=1024)
    
    if os.path.exists(output_dir + "/tokenized_tamil_CulturaX_dataset"):
        tokenized_dataset = DatasetDict.load_from_disk(output_dir + "/tokenized_tamil_CulturaX_dataset")
    else:
        dataset = load_dataset("uonlp/CulturaX", "ta")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        tokenized_dataset = DatasetDict({
            "train": dataset["train"].map(tokenize_function, batched=True),
            "validation": dataset["test"].map(tokenize_function, batched=True)
        })
        tokenized_dataset.save_to_disk(output_dir + "/tokenized_tamil_CulturaX_dataset")
    
    tokenizer.save_pretrained(output_dir)
    config = GPTNeoConfig(
        vocab_size=tokenizer.vocab_size,
        attention_types=[[['global', 'local'], 4]],
        num_layers=8,
        hidden_size=1024,
        max_position_embeddings=1024,
    )

    model = GPTNeoLightning(config, tokenizer, tokenized_dataset["train"], tokenized_dataset["validation"], learning_rate=1e-4)
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=500,
        default_root_dir=log_dir,
        enable_checkpointing=True,
        accumulate_grad_batches=8,
    )
    trainer.fit(model)
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/gptNeo-ta/")
    parser.add_argument("--log_dir", type=str, default="logs/gptNeo-ta/")
    args = parser.parse_args()
    train(args.output_dir, args.log_dir)
