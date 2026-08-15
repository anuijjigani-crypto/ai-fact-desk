#!/usr/bin/env bash
# Runs the full pipeline end to end: preprocess -> train all 3 models -> evaluate.
# Usage: bash run_pipeline.sh
set -e

echo "=== Step 1/3: Preprocessing data ==="
python src/data_preprocessing.py

echo "=== Step 2/3: Training models (bert, roberta, deberta) ==="
python src/train.py --model bert
python src/train.py --model roberta
python src/train.py --model deberta

echo "=== Step 3/3: Evaluating on test set ==="
python src/evaluate.py --model all

echo "Pipeline complete. Launch the app with: streamlit run app.py"
