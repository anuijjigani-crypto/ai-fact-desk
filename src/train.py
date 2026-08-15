"""
Step 2 of the pipeline: fine-tune BERT / RoBERTa / DeBERTa.

Features:
- Saves checkpoints every 100 training steps.
- Automatically resumes from the latest checkpoint.
- Saves the best model based on validation F1.
- Saves tokenizer with the best model.

Commands:

python src/train.py --model bert --epochs 1 --batch_size 16 --max_len 64
python src/train.py --model roberta --epochs 1 --batch_size 16 --max_len 64
python src/train.py --model deberta --epochs 1 --batch_size 16 --max_len 64
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.dataset import NewsDataset


# ---------------------------------------------------------
# Set random seed
# ---------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Find latest checkpoint
# ---------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = glob.glob(
        os.path.join(checkpoint_dir, "checkpoint-*")
    )

    if not checkpoints:
        return None

    def checkpoint_number(path):
        match = re.search(r"checkpoint-(\d+)", path)

        if match:
            return int(match.group(1))

        return -1

    checkpoints.sort(key=checkpoint_number)

    return checkpoints[-1]


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model, dataloader, device):

    model.eval()

    all_preds = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            labels = batch["labels"].to(device)

            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k != "labels"
            }

            outputs = model(
                **inputs,
                labels=labels
            )

            total_loss += outputs.loss.item()

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    metrics = {

        "loss":
            total_loss / len(dataloader),

        "accuracy":
            accuracy_score(
                all_labels,
                all_preds
            ),

        "precision":
            precision_score(
                all_labels,
                all_preds,
                zero_division=0
            ),

        "recall":
            recall_score(
                all_labels,
                all_preds,
                zero_division=0
            ),

        "f1":
            f1_score(
                all_labels,
                all_preds,
                zero_division=0
            ),
    }

    return metrics


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(config.MODEL_REGISTRY.keys())
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=config.MAX_LEN
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Seed
    # -----------------------------------------------------

    set_seed(config.SEED)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------

    model_info = config.MODEL_REGISTRY[args.model]

    hf_name = model_info["hf_name"]

    save_dir = model_info["save_dir"]

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # Checkpoint directory
    checkpoint_dir = os.path.join(
        save_dir,
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    print(
        f"Loading tokenizer/model for {hf_name} ..."
    )

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(

        hf_name,

        num_labels=2,

        id2label=config.ID2LABEL,

        label2id=config.LABEL2ID

    ).to(device)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    train_df = pd.read_csv(
        config.TRAIN_CSV
    )

    val_df = pd.read_csv(
        config.VAL_CSV
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    # -----------------------------------------------------
    # Dataset objects
    # -----------------------------------------------------

    train_ds = NewsDataset(

        train_df["content"],

        train_df["label"],

        tokenizer,

        args.max_len

    )

    val_ds = NewsDataset(

        val_df["content"],

        val_df["label"],

        tokenizer,

        args.max_len

    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(

        train_ds,

        batch_size=args.batch_size,

        shuffle=True

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=args.batch_size

    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=args.lr

    )

    # -----------------------------------------------------
    # Total training steps
    # -----------------------------------------------------

    total_steps = (
        len(train_loader)
        * args.epochs
    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=int(
            0.1 * total_steps
        ),

        num_training_steps=total_steps

    )

    # -----------------------------------------------------
    # Check for previous checkpoint
    # -----------------------------------------------------

    latest_checkpoint = find_latest_checkpoint(
        checkpoint_dir
    )

    start_epoch = 1

    global_step = 0

    best_f1 = 0.0

    if latest_checkpoint:

        print(
            "\nFound checkpoint:"
        )

        print(
            latest_checkpoint
        )

        print(
            "Resuming training..."
        )

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            latest_checkpoint
        ).to(device)

        # Load optimizer
        optimizer_path = os.path.join(
            latest_checkpoint,
            "optimizer.pt"
        )

        if os.path.exists(
            optimizer_path
        ):

            optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location=device
                )
            )

        # Load scheduler
        scheduler_path = os.path.join(
            latest_checkpoint,
            "scheduler.pt"
        )

        if os.path.exists(
            scheduler_path
        ):

            scheduler.load_state_dict(
                torch.load(
                    scheduler_path,
                    map_location=device
                )
            )

        # Load training information
        state_path = os.path.join(
            latest_checkpoint,
            "training_state.pt"
        )

        if os.path.exists(
            state_path
        ):

            state = torch.load(
                state_path,
                map_location=device
            )

            start_epoch = state.get(
                "epoch",
                1
            )

            global_step = state.get(
                "global_step",
                0
            )

            best_f1 = state.get(
                "best_f1",
                0.0
            )

        print(
            f"Resuming from epoch {start_epoch}"
        )

        print(
            f"Global step: {global_step}"
        )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs + 1
    ):

        model.train()

        running_loss = 0.0

        progress = tqdm(

            train_loader,

            desc=(
                f"[{args.model}] "
                f"Epoch {epoch}/{args.epochs}"
            )

        )

        for batch in progress:

            # -------------------------------------------------
            # Clear gradients
            # -------------------------------------------------

            optimizer.zero_grad()

            # -------------------------------------------------
            # Move labels to device
            # -------------------------------------------------

            labels = batch[
                "labels"
            ].to(device)

            # -------------------------------------------------
            # Prepare inputs
            # -------------------------------------------------

            inputs = {

                k: v.to(device)

                for k, v in batch.items()

                if k != "labels"

            }

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            outputs = model(

                **inputs,

                labels=labels

            )

            loss = outputs.loss

            # -------------------------------------------------
            # Backward pass
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0

            )

            # -------------------------------------------------
            # Update model
            # -------------------------------------------------

            optimizer.step()

            scheduler.step()

            # -------------------------------------------------
            # Update counters
            # -------------------------------------------------

            global_step += 1

            running_loss += loss.item()

            # -------------------------------------------------
            # Show progress
            # -------------------------------------------------

            progress.set_postfix(

                loss=(
                    running_loss
                    / (progress.n + 1)
                )

            )

            # -------------------------------------------------
            # Save checkpoint every 100 steps
            # -------------------------------------------------

            if global_step % 100 == 0:

                checkpoint_path = os.path.join(

                    checkpoint_dir,

                    f"checkpoint-{global_step}"

                )

                os.makedirs(

                    checkpoint_path,

                    exist_ok=True

                )

                print(
                    f"\nSaving checkpoint at step {global_step}..."
                )

                # Save model
                model.save_pretrained(
                    checkpoint_path
                )

                # Save optimizer
                torch.save(

                    optimizer.state_dict(),

                    os.path.join(
                        checkpoint_path,
                        "optimizer.pt"
                    )

                )

                # Save scheduler
                torch.save(

                    scheduler.state_dict(),

                    os.path.join(
                        checkpoint_path,
                        "scheduler.pt"
                    )

                )

                # Save tokenizer
                tokenizer.save_pretrained(
                    checkpoint_path
                )

                # Save training state
                torch.save(

                    {

                        "epoch": epoch,

                        "global_step":
                            global_step,

                        "best_f1":
                            best_f1

                    },

                    os.path.join(

                        checkpoint_path,

                        "training_state.pt"

                    )

                )

                print(
                    f"Checkpoint saved: {checkpoint_path}"
                )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        val_metrics = evaluate(

            model,

            val_loader,

            device

        )

        print(
            f"\nEpoch {epoch} validation:"
        )

        print(
            val_metrics
        )

        # -----------------------------------------------------
        # Save best model
        # -----------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            model.save_pretrained(
                save_dir
            )

            tokenizer.save_pretrained(
                save_dir
            )

            print(
                f"\nNew best model saved to:"
            )

            print(
                save_dir
            )

            print(
                f"Best F1: {best_f1:.4f}"
            )

    # -----------------------------------------------------
    # Training complete
    # -----------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        f"Training complete for {args.model}"
    )

    print(
        f"Best validation F1: {best_f1:.4f}"
    )

    print(
        f"Model saved to: {save_dir}"
    )

    print(
        "======================================"
    )


# ---------------------------------------------------------
# Run main
# ---------------------------------------------------------

if __name__ == "__main__":
    main()