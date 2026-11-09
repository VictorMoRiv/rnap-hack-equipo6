from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)
ds = load_dataset("AmazonScience/massive", "es-ES")
for split in ["train", "validation", "test"]:
    ds[split].to_json(f"data/massive_es-ES_{split}.jsonl", force_ascii=False)
    print(split, len(ds[split]), "ejemplos guardados")