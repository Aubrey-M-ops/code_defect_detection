import json
from typing import List, Tuple
import os

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append(obj)
    return data

def load_splits(data_dir: str):
    train = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    valid = load_jsonl(os.path.join(data_dir, "valid.jsonl"))
    test  = load_jsonl(os.path.join(data_dir, "test.jsonl"))
    return train, valid, test

def extract_x_y(data):
    texts = [d["func"] for d in data]
    labels = [int(d["target"]) for d in data]
    return texts, labels
