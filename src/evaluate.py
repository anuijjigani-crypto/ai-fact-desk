"""
Step 3 of the pipeline: evaluate one or all fine-tuned models on the
held-out test split and print/save a report + confusion matrix.

Usage:
    python src/evaluate.py --model bert
    python src/evaluate.py --model all
"""
import argparse
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import NewsDataset


def evaluate_model(model_key, device, batch_size=16, max_len=256):
    save_dir = config.MODEL_REGISTRY[model_key]["save_dir"]
    if not os.path.exists(save_dir):
        print(f"[skip] No fine-tuned model found at {save_dir}. Train it first.")
        return

    print(f"\n=== Evaluating {model_key} ===")
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    model.eval()

    test_df = pd.read_csv(config.TEST_CSV)
    test_ds = NewsDataset(test_df["content"], test_df["label"], tokenizer, max_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds, target_names=["FAKE", "REAL"], digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(report)
    print("Confusion matrix [rows=true, cols=pred] (FAKE, REAL):")
    print(cm)

    report_path = os.path.join(save_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        f.write("\n\nConfusion matrix (FAKE, REAL):\n")
        f.write(str(cm))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(config.MODEL_REGISTRY.keys()) + ["all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = list(config.MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for key in keys:
        evaluate_model(key, device)
