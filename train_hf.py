from ast import arg
from transformers import GPTNeoConfig, GPTNeoForCausalLM, LlamaForCausalLM,LlamaConfig , Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
import argparse
import os
import json
import evaluate
import warnings
import torch
import wandb
import torch
import torch.nn.functional as F
import numpy as np

# os.environ["WANDB_PROJECT"] = "Kurunkathai" 
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

def train_val_split(dataset, split_ratio=0.9):
    dataset = dataset["train"].train_test_split(test_size=1-split_ratio, seed=42)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

perplexity_eval = evaluate.load("perplexity", module_type="metric")

def train(training_config):
    wandb.init(project="Kurunkathai")
    
    odir = training_config["output_dir"]
    output_dir =os.path.join("models",odir)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(training_config["tokenizer_name"])
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"Selecting device {device}")
    generated_table = wandb.Table(columns=["Run ID", "Eval Step", "Generated Text", "Reference Text"])
    sample_text = "செல்வன் என்ற சிறுவன் பள்ளிக்கு செல்ல விரும்பாமல்"    
    sample_text_ids = tokenizer(sample_text, return_tensors="pt").to(device)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="longest", max_length=1024)
   
    if training_config["dataset"] == "CulturaX":
        dataset = load_dataset("uonlp/CulturaX", "ta")
        dataset = dataset["train"].train_test_split(test_size=training_config["default_train_test_split"], seed=42)
        tokenized_dataset = DatasetDict({"train": dataset["train"].map(tokenize_function, batched=True),"validation": dataset["test"].map(tokenize_function, batched=True)})
    else:
        dataset = load_dataset(training_config["dataset"])
        if "validation" not in dataset.keys():
            warnings.warn("Validation set not found. Splitting the training set to create a validation set.")
            dataset_split = dataset["train"].train_test_split(test_size=training_config["default_train_test_split"], seed=42)
            dataset = DatasetDict({"train": dataset_split["train"], "validation": dataset_split["test"]})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenizer.save_pretrained(output_dir)

    if training_config.get("model_path_dict",None) is not None:
        causalLM = AutoModelForCausalLM.from_pretrained(**training_config["model_path_dict"])
        if causalLM.config.vocab_size != tokenizer.vocab_size:
            warnings.warn("Resizing token embeddings to match the tokenizer's vocab size.")
            causalLM.resize_token_embeddings(tokenizer.vocab_size)
            causalLM.config.vocab_size = tokenizer.vocab_size
        causalLM.config.pad_token_id = tokenizer.pad_token_id
    else:
        if training_config.get("GPTNeo",True):
            config = GPTNeoConfig(
                vocab_size=tokenizer.vocab_size,
                attention_types=[[['global', 'local'], 4]],
                num_layers=8,
                hidden_size=1024,
                max_position_embeddings=1024,
            )
            causalLM = GPTNeoForCausalLM(config)
        else:
            config = LlamaConfig(vocab_size=tokenizer.vocab_size, attention_bias = False,attention_dropout= 0.1, bos_token_id= tokenizer.bos_token_id,  eos_token_id= tokenizer.eos_token_id, hidden_act= "silu", hidden_size= 512,  initializer_range=0.041666666666666664, intermediate_size= 1536, is_llama_config= True, max_position_embeddings= 512, model_type= "llama",  num_attention_heads= 8,  num_hidden_layers= 4, num_key_value_heads= 4,  pretraining_tp= 1,  rms_norm_eps= 1e-05,  rope_interleaved= False, rope_scaling=None ,rope_theta=100000)
            causalLM = LlamaForCausalLM(config)
    eval_batch_mult = 1#4 if training_config.get("GPTNeo", True) else 1
   
    if "lora_config" in training_config:
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **training_config["lora_config"])
        causalLM = get_peft_model(causalLM, lora_config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)
    
    class CustomTrainer(Trainer):
        def __init__(self, *args, eval_datasets, **kwargs):
            kwargs["eval_dataset"] = eval_datasets[0] if eval_datasets else None
            super().__init__(*args, **kwargs)
            self.eval_datasets = eval_datasets if eval_datasets is not None else [self.eval_dataset]

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval",  **kwargs):
            generated_text = tokenizer.decode(causalLM.generate(**sample_text_ids)[-1], skip_special_tokens=True)            
            generated_table.add_data(wandb.run.id, trainer.state.global_step, sample_text, generated_text)
            wandb.log({"generated_text": generated_table})
            results = {}

            datasets = self.eval_datasets  # Use stored datasets
            
            for i, dataset in enumerate(datasets):
                output = super().evaluate(
                    eval_dataset=dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_dataset_{i}",
                    **kwargs
                )
                eval_loss = output.get(f"{metric_key_prefix}_dataset_{i}_loss")
                perplexity = torch.exp(torch.tensor(eval_loss)).item()
                output[f"{metric_key_prefix}_dataset_{i}_perplexity"] = perplexity
                wandb.log({f"perplexity_dataset_{i}": perplexity})
                results.update(output)
            return results

    if "hq_dataset" in training_config:
        eval_dataset = [tokenized_dataset["validation"], load_dataset(training_config["hq_dataset"])["validation"].map(tokenize_function, batched=True)]
    else:
        eval_dataset = [tokenized_dataset["validation"]]
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=training_config["val_check_interval"],
        save_strategy="steps",
        save_steps = training_config["val_check_interval"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size= eval_batch_mult*training_config["batch_size"],
        learning_rate = training_config["learning_rate"],
        num_train_epochs=training_config["max_epochs"],
        save_total_limit=training_config["num_checkpoints_to_keep"],
        logging_steps=training_config["log_every_n_steps"],
        do_train=True,
        do_eval=True,
        gradient_accumulation_steps=training_config["accumulate_grad_batches"],
        weight_decay = training_config["weight_decay"],
        fp16=False,
        push_to_hub=training_config["push_to_hub"],
        report_to="wandb",
        max_grad_norm = training_config["max_grad_norm"]
    )

    if "adam_beta1" in training_config:
        training_args.adam_beta1 = training_config["adam_beta1"]
    if "adam_beta2" in training_config:
        training_args.adam_beta2 = training_config["adam_beta2"]

    trainer = CustomTrainer(
        model=causalLM,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_datasets=eval_dataset,
        eval_dataset=eval_dataset[0],
        data_collator=data_collator)
    
    # Train the model
    causalLM.push_to_hub(odir, commit_message="Before training")
    trainer.evaluate() 
    trainer.train()

    # Save final model and tokenizer
    causalLM.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    causalLM.push_to_hub(odir, commit_message="Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--config", type=str, default=".config/defaults.json")
    args = parser.parse_args()
    training_config = json.load(open(args.config,"r"))
    train(training_config)
