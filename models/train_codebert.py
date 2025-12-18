import os
import sys
import json

# Add utils to path (must be before importing utils)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_classification_metrics
from utils.log import write_to_log
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
from collections import Counter

DATA_FILES = {
    "train": "data/raw/train.jsonl",
    "validation": "data/raw/validation.jsonl",
    "test": "data/raw/test.jsonl",
}

MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "ckpt_codebert"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(max_length=256, num_epochs=3, batch_size=16, lr=2e-5):
    # 1) load data
    ds = load_dataset("json", data_files=DATA_FILES)

    # 2) tokenizer
    # Automatically load the tokenizer matching with the current model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        result = tok(
            examples["func"],           # Input text: function code
            truncation=True,             # Truncate sequences longer than max_length
            padding="max_length",        # Pad sequences to max_length
            max_length=max_length        # Maximum sequence length
        )

        # Log tokenization result for inspection
        # NOTE: temp log
        log_content = json.dumps({
            "input_ids_shape": str(np.array(result["input_ids"]).shape),
            "attention_mask_shape": str(np.array(result["attention_mask"]).shape),
            "sample_input_ids": result["input_ids"][0][:20] if result["input_ids"] else [],
            "sample_attention_mask": result["attention_mask"][0][:20] if result["attention_mask"] else []
        }, indent=2)
        write_to_log(log_content, "preprocess_tokenizer.log")

        return result

    # ds_tok stores the meta data of processed dataset
    ds_tok = ds.map(preprocess, batched=True)

    # HuggingFace Trainer uses "labels" instead of "target"
    ds_tok = ds_tok.rename_column("target", "labels")
    ds_tok.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    print(ds_tok)

    # 3) class weights (handle class imbalance, because 0 is much more than 1)
    label_list = np.array(ds_tok["train"]["labels"])
    counter = Counter(label_list)
    print("Label distribution:", counter)
    # Assume labels are 0/1
    n0, n1 = counter[0], counter[1]
    # Give minority class higher weight
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

    # Wrap with custom loss using class weights
    class WeightedModel(nn.Module):
        def __init__(self, model, class_weights):
            super().__init__()
            self.model = model
            self.loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights)
            )

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            # Remove num_items_in_batch if present (not needed by the model)
            kwargs.pop('num_items_in_batch', None)
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
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        result = compute_classification_metrics(labels, preds)
        return result

    # 6) training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",   # or mcc
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
