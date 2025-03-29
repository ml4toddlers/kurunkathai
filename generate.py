from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def generate_text(model_path, tokenizer_path, prompt, max_new_tokens=150):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens= max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, attention_mask=input_ids.ne(tokenizer.pad_token_id))
    return tokenizer.decode(output[0], skip_special_tokens=True)

def write_as_markdown(prompt, output, filename):
    with open(filename, "a") as f:
        f.write(f"\n# {prompt}\n\n")
        f.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--output", type=str, default="outputs/output.md")
    args = parser.parse_args()
    write_as_markdown(args.prompt, generate_text(args.model_path, args.tokenizer_path, args.prompt, args.max_new_tokens), args.output)
