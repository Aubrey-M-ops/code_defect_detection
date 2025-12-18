from datasets import load_dataset
import os, json

ds = load_dataset("semeru/code-code-DefectDetection")

os.makedirs("data/raw", exist_ok=True)

for split in ["train", "validation", "test"]:
    path = f"data/raw/{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in ds[split]:
            f.write(json.dumps({
                "func": row["func"],
                "target": int(row["target"]),
                "idx": int(row["idx"])
            }) + "\n")
    print("saved:", path)
