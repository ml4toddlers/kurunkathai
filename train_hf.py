from transformers import GPTNeoConfig, GPTNeoForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
import argparse
import os
import json
import evaluate
import warnings

os.environ["WANDB_PROJECT"] = "Kurunkathai" 
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
def train_val_split(dataset, split_ratio=0.9):
    dataset = dataset["train"].train_test_split(test_size=1-split_ratio, seed=42)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

perplexity_eval = evaluate.load("perplexity", module_type="metric")

def train(training_config):
    odir = training_config["output_dir"]
    output_dir =os.path.join("models",odir)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(training_config["tokenizer_name"])

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="longest", max_length=1024)
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        mean_ppl = perplexity_eval.compute(predictions = decoded_preds, model_id = "gpt_neo", add_start_token=False, batch_size = training_config["batch_size"])["mean_perplexity"]
        return {"perplexity": mean_ppl}
      
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
    else:
        config = GPTNeoConfig(
            vocab_size=tokenizer.vocab_size,
            attention_types=[[['global', 'local'], 4]],
            num_layers=8,
            hidden_size=1024,
            max_position_embeddings=1024,
        )
        causalLM = GPTNeoForCausalLM(config)
    
    if "lora_config" in training_config:
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **training_config["lora_config"])
        causalLM = get_peft_model(causalLM, lora_config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=training_config["val_check_interval"],
        save_strategy="steps",
        save_steps = training_config["val_check_interval"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        learning_rate = training_config["learning_rate"],
        num_train_epochs=training_config["max_epochs"],
        save_total_limit=training_config["num_checkpoints_to_keep"],
        logging_steps=training_config["log_every_n_steps"],
        do_train=True,
        do_eval=True,
        gradient_accumulation_steps=training_config["accumulate_grad_batches"],
        weight_decay = training_config["weight_decay"],
        fp16=True,
        push_to_hub=training_config["push_to_hub"],
        report_to="wandb"
    )

    trainer = Trainer(
        model=causalLM,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.evaluate() 
    trainer.train()

    # Save final model and tokenizer
    causalLM.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--config", type=str, default=".config/defaults.json")
    args = parser.parse_args()
    training_config = json.load(open(args.config,"r"))
    train(training_config)
