import argparse
import subprocess
from tqdm import tqdm
import time

def run_subproceses(ds, split, start_index, end_index):
    num_batches_per_process = 5
    batch_size_per_process = 500
    batch_size = num_batches_per_process * batch_size_per_process
    for i in tqdm(range(start_index, end_index, batch_size),desc=f"Translating {split}_{start_index}_{end_index}"):
        args = ["python", "TinyStories/translate_batches.py", f"--ds={ds}", f"--split={split}", f"--start_index={i}", f"--num_batches={num_batches_per_process}", f"--batch_size={batch_size_per_process}"]
        while True: 
            try:
                result = subprocess.run(args, check=True)
                print(f"Successfully completed: {args}")
                break  
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running {args}: {e}")
                print("Retrying...")
                time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)
    args = parser.parse_args().__dict__
    run_subproceses(**args)