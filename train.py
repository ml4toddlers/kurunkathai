import json
import os
import math
import argparse
import torch
import pytorch_lightning as pl
from transformers import (GPTNeoConfig, GPTNeoForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, AdamW, get_linear_schedule_with_warmup)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import warnings
import datetime
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')
warnings.simplefilter("always", UserWarning)  
warnings.filterwarnings("ignore", category=UserWarning, module=".*")  

class GPTNeoLightning(pl.LightningModule):
    def __init__(self, causalLM, tokenizer, train_dataset, val_dataset, training_config):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.model = causalLM
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = training_config["learning_rate"]
        self.weight_decay = training_config["weight_decay"]
        self.batch_size = training_config["batch_size"]
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.num_cpus = os.cpu_count()
        self.name = training_config["output_dir"]
        self.num_vals = 0
    
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
        if batch_idx == 0:  
            test = "ஏன் இவ்வாறு செய்கிறீர்கள் என்று எழிலன் கேட்க "
            inputs = self.tokenizer(test, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs)
            decoded_output = self.tokenizer.decode(output[-1], skip_special_tokens=True)
            print(decoded_output)
            self.logger.log_text(f"sample_output_{self.name}", columns =["epoch", "output"],data=[[self.num_vals,[decoded_output]]])
            self.num_vals += 1
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        num_training_steps=len(self.train_dataloader()) * self.trainer.max_epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps )
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator, num_workers=self.num_cpus)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.data_collator, num_workers=self.num_cpus)

           
def train(training_config):
    odir = training_config["output_dir"]
    output_dir =os.path.join("models",odir)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(training_config["tokenizer_name"])
    wandb_logger = WandbLogger(project="Kurunkathai", name=f"{odir}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_val_loss_{val_loss:.2f}",
        dirpath=output_dir,
        save_top_k=1,
        mode="min",
        save_last=True,
        every_n_epochs=5
    )

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="longest", max_length = 1024)
    
    if training_config["dataset"] == "CulturaX":
        dataset = load_dataset("uonlp/CulturaX", "ta")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        tokenized_dataset = DatasetDict({"train": dataset["train"].map(tokenize_function, batched=True),"validation": dataset["test"].map(tokenize_function, batched=True)})
    else:
        dataset = load_dataset(training_config["dataset"])
        if "validation" not in dataset.keys():
            warnings.warn("Validation set not found. Splitting the training set to create a validation set.")
            dataset_split = dataset["train"].train_test_split(test_size=0.025, seed=42)
            dataset = DatasetDict({"train": dataset_split["train"], "validation": dataset_split["test"]})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])        
    
    tokenizer.save_pretrained(output_dir)
    if training_config.get("model_path_dict",None) is not None:
        causalLM = GPTNeoForCausalLM.from_pretrained(**training_config["model_path_dict"])
    else:
        config = GPTNeoConfig(
            vocab_size=tokenizer.vocab_size,
            attention_types=[[['global', 'local'], 4]],
            num_layers=8,
            hidden_size=1024,
            max_position_embeddings=1024,
        )
        causalLM = GPTNeoForCausalLM(config)

    model = GPTNeoLightning(causalLM, tokenizer, tokenized_dataset["train"], tokenized_dataset["validation"], training_config)
    
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=training_config["log_every_n_steps"],
        default_root_dir=output_dir,
        enable_checkpointing=True,
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        val_check_interval=training_config.get("val_check_interval",1.0),
    )
    if training_config.get("push_to_hub", False):
        class PushToHubCallback(pl.Callback):
            def on_epoch_end(self, trainer, pl_module):
                if trainer.current_epoch % 5 == 0:
                    pl_module.model.push_to_hub(training_config["output_dir"])
        
        trainer.callbacks.append(PushToHubCallback())
    
    trainer.fit(model)
    model.model.save_pretrained(output_dir)
    if training_config.get("push_to_hub", False):
        model.model.push_to_hub(training_config["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--config", type=str, default=".config/defaults.json")
    args = parser.parse_args()
    training_config = json.load(open(args.config,"r"))
    train(training_config)
