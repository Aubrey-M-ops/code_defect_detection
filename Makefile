.PHONY: help data install train clean test

data:
	@echo "Loading dataset..."
	python3 data/dataset_loading.py
	@echo "Dataset loaded successfully!"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed!"

train-baseline:
	@echo "Training CodeBERT model..."
	python3 models/train_codebert.py

train-ast:
	@echo "Training CodeBERT+AST model..."
	python3 models/train_codebert_ast.py


clean-data:
	@echo "Cleaning data directory..."
	rm -rf data/raw/*.jsonl
	rm -rf data/processed/*
	@echo "Data cleaned!"
