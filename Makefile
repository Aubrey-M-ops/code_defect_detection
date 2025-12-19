.PHONY: help data install train clean test

# Use Python from venv if available, otherwise system python3
PYTHON := $(shell if [ -d venv ]; then echo ./venv/bin/python; else echo python3; fi)
PIP := $(shell if [ -d venv ]; then echo ./venv/bin/pip; else echo pip; fi)

data:
	@echo "Loading dataset..."
	$(PYTHON) data/dataset_loading.py
	@echo "Dataset loaded successfully!"

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed!"

train-baseline:
	@echo "Training CodeBERT model..."
	$(PYTHON) models/train_codebert.py

train-ast:
	@echo "Training CodeBERT+AST model..."
	$(PYTHON) models/train_codebert_ast.py

ablation:
	@echo "Running ablation study on input length..."
	bash experiments/run_ablation_length.sh
	@echo "Ablation study completed!"

clean-data:
	@echo "Cleaning data directory..."
	rm -rf data/raw/*.jsonl
	rm -rf data/processed/*
	@echo "Data cleaned!"
