"""
Central configuration for the Fake News Detection Agent project.
Edit values here rather than hard-coding them across scripts.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FAKE_CSV = os.path.join(RAW_DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(RAW_DATA_DIR, "True.csv")

TRAIN_CSV = os.path.join(PROCESSED_DATA_DIR, "train.csv")
VAL_CSV = os.path.join(PROCESSED_DATA_DIR, "val.csv")
TEST_CSV = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# 0 = FAKE, 1 = REAL  (matches the Kaggle "Fake and Real News" dataset)
LABEL2ID = {"FAKE": 0, "REAL": 1}
ID2LABEL = {0: "FAKE", 1: "REAL"}

# ---------------------------------------------------------------------------
# Models the agent is built on. "key" is used for folder names / UI labels.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bert": {
        "hf_name": "bert-base-uncased",
        "display_name": "BERT (bert-base-uncased)",
        "save_dir": os.path.join(MODELS_DIR, "bert"),
    },
    "roberta": {
        "hf_name": "roberta-base",
        "display_name": "RoBERTa (roberta-base)",
        "save_dir": os.path.join(MODELS_DIR, "roberta"),
    },
    "deberta": {
        "hf_name": "microsoft/deberta-v3-base",
        "display_name": "DeBERTa (microsoft/deberta-v3-base)",
        "save_dir": os.path.join(MODELS_DIR, "deberta"),
    },
}

# Ensemble weights (equal by default). Tune after evaluating each model.
ENSEMBLE_WEIGHTS = {"bert": 1.0, "roberta": 1.0, "deberta": 1.0}

# ---------------------------------------------------------------------------
# Training hyperparameters (defaults, override via CLI args in train.py)
# ---------------------------------------------------------------------------
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# ---------------------------------------------------------------------------
# Optional: external fact-check cross-referencing in the agent layer.
# Leave NEWSAPI_KEY empty to skip web cross-referencing entirely — the
# agent still works fine on model predictions alone.
# ---------------------------------------------------------------------------
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
