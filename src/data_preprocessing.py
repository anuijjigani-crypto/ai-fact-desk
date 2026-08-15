"""
Step 1 of the pipeline.

Loads the raw Kaggle "Fake and Real News" dataset (Fake.csv + True.csv),
cleans the text, merges title + body, and produces stratified
train / val / test CSV splits under data/processed/.

Usage:
    python src/data_preprocessing.py
"""
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9\s.,!?'\"-]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light-touch cleaning. We keep punctuation since transformer
    tokenizers use it, but strip URLs, source tags (e.g. 'Reuters -')
    and junk whitespace, which are strong shortcut-features that hurt
    generalization if left in."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    # Many "real" articles in this dataset start with "CITY (Reuters) -"
    # which is a trivial shortcut for the model to memorize. Strip it.
    text = re.sub(r"^[A-Z\s,]+\(Reuters\)\s*-\s*", "", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def load_and_merge() -> pd.DataFrame:
    if not (os.path.exists(config.FAKE_CSV) and os.path.exists(config.TRUE_CSV)):
        raise FileNotFoundError(
            "Could not find Fake.csv / True.csv in data/raw/.\n"
            "Download the dataset from:\n"
            "  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "and place Fake.csv and True.csv inside data/raw/."
        )

    fake_df = pd.read_csv(config.FAKE_CSV)
    true_df = pd.read_csv(config.TRUE_CSV)

    fake_df["label"] = config.LABEL2ID["FAKE"]
    true_df["label"] = config.LABEL2ID["REAL"]

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text; both carry signal for fake-news detection.
    df["title"] = df.get("title", "").fillna("")
    df["text"] = df.get("text", "").fillna("")
    df["content"] = (df["title"] + ". " + df["text"]).str.strip()

    df["content"] = df["content"].apply(clean_text)

    # Drop empties / near-empties and exact duplicates.
    df = df[df["content"].str.split().apply(len) > 10]
    df = df.drop_duplicates(subset=["content"])
    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    return df[["content", "label"]]


def split_and_save(df: pd.DataFrame) -> None:
    train_df, temp_df = train_test_split(
        df,
        test_size=(config.VAL_SIZE + config.TEST_SIZE),
        stratify=df["label"],
        random_state=config.SEED,
    )
    relative_test = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=config.SEED,
    )

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Label balance (train):")
    print(train_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    merged = load_and_merge()
    print(f"Loaded {len(merged)} cleaned articles.")
    split_and_save(merged)
    print("Done. Processed CSVs written to data/processed/")
