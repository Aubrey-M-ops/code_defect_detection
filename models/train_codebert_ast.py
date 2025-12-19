import os
import sys
import json
from datetime import datetime

# Add utils to path (must be before importing utils)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_classification_metrics
from utils.log import write_to_log
from utils.ast_utils import make_augmented_text
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, TrainerCallback)
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


class TrainingLogCallback(TrainerCallback):
    """Custom callback to log training metrics to file"""

    def __init__(self, log_file="training_metrics.log"):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs is not None:
            log_content = json.dumps(logs, indent=2)
            write_to_log(log_content, self.log_file, append=True)


def main(max_length=256, num_epochs=3, batch_size=16, lr=2e-5):
    # 1) load data
    ds = load_dataset("json", data_files=DATA_FILES)

    # 2) tokenizer
    # Automatically load the tokenizer matching with the current model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        # NOTE: !!!!!!!!!!!! add AST Sequence before code !!!!!!!!!!!!
        new_texts = [make_augmented_text(code) for code in examples["func"]]

        return tok(
            new_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # ds_tok stores the meta data of processed dataset
    ds_tok = ds.map(preprocess, batched=True)

    # HuggingFace Trainer uses "labels" instead of "target"
    ds_tok = ds_tok.rename_column("target", "labels")
    ds_tok.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

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
        #Gradient accumulation => a larger effective batch size.
        gradient_accumulation_steps=2,

        # warmup + scheduler + gradient clipping (increase stability)
        # for BERT/CodeBERT
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_grad_norm=1.0,

        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[TrainingLogCallback(log_file="training_metrics.log")]
    )

    trainer.train()
    print("Eval on test:")
    test_results = trainer.evaluate(ds_tok["test"])
    print(test_results)

    ################################## PRINT TO LOG >>> ##################################

    # Save training args and results to log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_log_file = f"result_ast_{timestamp}.log"


    # Create result object
    result_object = {
        "timestamp": timestamp,
        "training_results": test_results
    }

    # Write to log file
    log_content = json.dumps(result_object, indent=2, ensure_ascii=False)
    write_to_log(log_content, result_log_file, append=False)
    print(f"\nTraining results saved to log/{result_log_file}")
    ################################################################################### 


if __name__ == "__main__":
    main()
