from transformers import GPTNeoConfig, GPTNeoForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import torch
import math

def train_val_split(dataset, split_ratio=0.9):
    dataset = dataset["train"].train_test_split(test_size=1-split_ratio, seed=42)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})


def tokenize_function(batch):
    if "text" in batch:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    else:
        concatenated_text = [inp + " " + tgt for inp, tgt in zip(batch["inputs"], batch["targets"])]
        return tokenizer(concatenated_text, truncation=True, padding="max_length", max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = logits.mean()  
    perplexity = math.exp(loss)  
    return {"loss": loss, "perplexity": perplexity}

dataset = load_dataset("uonlp/CulturaX", "ta")
tokenizer = AutoTokenizer.from_pretrained("abhinand/tamil-llama-7b-instruct-v0.2")


dataset = train_val_split(dataset)
dataset_high_quality =  DatasetDict({"validation":load_dataset("aitamilnadu/tamil_stories")["train"]})

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_hq_dataset = dataset_high_quality.map(tokenize_function, batched=True)

tokenizer.save_pretrained("./GPTNeo_1024_12_ta")
config = GPTNeoConfig(
    vocab_size=tokenizer.vocab_size,
    attention_types=[[['global', 'local'], 6]],
    num_layers=12,
    hidden_size=1024,
    max_position_embeddings=2048,
)

model = GPTNeoForCausalLM(config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False)

class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, **kwargs):
        # First, evaluate on the primary validation dataset
        result_1 = super().evaluate(eval_dataset=tokenized_dataset["validation"], **kwargs)
        
        # Then, evaluate on the second validation dataset
        result_2 = super().evaluate(eval_dataset=tokenized_hq_dataset["validation"], **kwargs)
        
        # Combine the results or print them separately
        print("\nEvaluation on validation dataset 1:", result_1)
        print("\nEvaluation on validation dataset 2:", result_2)
        
        return {
            "eval_loss_CulturaX": result_1["loss"],
            "eval_loss_HQ_Stories": result_2["loss"],

            "perpexlity_CulturaX": result_1["loss"],
            "perpexlity_HQ_Stories": result_2["loss"]
        }

training_args = TrainingArguments(
    output_dir="./GPTNeo_1024_8_ta",
    eval_strategy="steps",
    eval_steps=0.01,
    save_strategy="steps",
    save_steps = 0.01,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    learning_rate = 1e-5,
    max_grad_norm=1.0,
    num_train_epochs=1,
    save_total_limit=4,
    logging_dir="./logs_gptneo",
    logging_steps=500,
    do_train=True,
    do_eval=True,
    bf16=torch.cuda.is_available(),
    gradient_accumulation_steps=8,
    fp16=False,
    push_to_hub=True
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=None,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save final model and tokenizer
model.save_pretrained("./GPTNeo_1024_8_ta")
tokenizer.save_pretrained("./GPTNeo_1024_8_ta")
