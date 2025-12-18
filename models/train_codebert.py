# models/train_codebert.py
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
import evaluate
from collections import Counter
import os

DATA_FILES = {
    "train": "data/raw/train.jsonl",
    "validation": "data/raw/valid.jsonl",
    "test": "data/raw/test.jsonl",
}

MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "ckpt_codebert"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(max_length=256, num_epochs=3, batch_size=16, lr=2e-5):
    # 1) load data
    ds = load_dataset("json", data_files=DATA_FILES)

    # 2) tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        return tok(
            examples["func"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    ds_tok = ds.map(preprocess, batched=True)

    # HuggingFace Trainer 需要这些字段
    ds_tok = ds_tok.rename_column("target", "labels")
    ds_tok.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # 3) class weights（处理类别不平衡）
    label_list = ds_tok["train"]["labels"].numpy()
    counter = Counter(label_list)
    print("Label distribution:", counter)
    # 假设 labels=0/1
    n0, n1 = counter[0], counter[1]
    # 让少数类有更高权重
    w0 = 1.0
    w1 = n0 / n1 if n1 > 0 else 1.0
    class_weights = np.array([w0, w1], dtype="float32")
    print("Class weights:", class_weights)

    # 4) model
    import torch
    import torch.nn as nn

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # 包一层自定义 loss，使用 class weight
    class WeightedModel(nn.Module):
        def __init__(self, model, class_weights):
            super().__init__()
            self.model = model
            self.loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights)
            )

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                **kwargs
            )
            logits = outputs.logits
            loss = None
            if labels is not None:
                loss = self.loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}

    model = WeightedModel(base_model, class_weights)

    # 5) metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")
    # MCC 没有现成的，用 sklearn
    from sklearn.metrics import matthews_corrcoef

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        result = {
            "accuracy": acc_metric.compute(
                predictions=preds, references=labels)["accuracy"],
            "f1": f1_metric.compute(
                predictions=preds, references=labels, average="binary")["f1"],
            "mcc": matthews_corrcoef(labels, preds),
        }
        return result

    # 6) training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",   # 或 mcc
        greater_is_better=True,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("Eval on test:")
    print(trainer.evaluate(ds_tok["test"]))

if __name__ == "__main__":
    main()
