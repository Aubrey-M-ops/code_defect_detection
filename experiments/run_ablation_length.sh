#!/bin/bash
set -e

# Ablation study on input length for CodeBERT+AST
python models/train_codebert.py --max_length 128 --output_file log/ablation_128.json
python models/train_codebert.py --max_length 256 --output_file log/ablation_256.json
python models/train_codebert.py --max_length 512 --output_file log/ablation_512.json
