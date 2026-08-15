# AI Fact Desk — Fake News Detection & Fact Verification Agent

An agent that reads a news article, runs it through three independently
fine-tuned transformer classifiers (**BERT**, **RoBERTa**, **DeBERTa**),
combines their votes into a single weighted verdict, explains *why* it
flagged the piece, and (optionally) cross-references it against live
headlines. Served through a Streamlit UI.

Dataset: [Fake and Real News Dataset (Kaggle, Clément Bisaillon)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 1. Project structure

```
fake-news-agent/
├── app.py                     # Streamlit UI (entry point)
├── config.py                  # All paths / hyperparameters / model registry
├── run_pipeline.sh            # Runs steps 2-5 below in one go
├── requirements.txt
├── data/
│   ├── raw/                   # <- put Fake.csv & True.csv here
│   └── processed/             # generated: train.csv / val.csv / test.csv
├── models/                    # generated: fine-tuned model weights per model
├── src/
│   ├── data_preprocessing.py  # Step 2
│   ├── dataset.py             # PyTorch Dataset (tokenization)
│   ├── train.py                # Step 3 (run once per model)
│   ├── evaluate.py            # Step 4
│   └── agent.py                # The ensemble "agent" (loads models, votes, explains)
└── notebooks/
    └── eda.ipynb              # optional exploratory data analysis
```

---

## 2. Why this counts as an "agent" and not just a classifier

`src/agent.py`'s `FakeNewsAgent` does more than call `model.predict()`:

1. **Orchestrates multiple tools** — loads whichever of BERT / RoBERTa /
   DeBERTa have been trained, and skips gracefully if one is missing.
2. **Aggregates & reasons** — combines each model's probability into a
   weighted ensemble verdict, and reports where the models agree/disagree.
3. **Explains itself** — extracts the sentences that most likely drove the
   decision and flags sensationalist language patterns.
4. **Can call external tools** — if you set a `NEWSAPI_KEY`, it cross-checks
   the article's topic against live headlines as a lightweight
   fact-verification signal (fully optional, degrades silently without it).
5. **Never hard-fails** — if no model is trained yet, it falls back to a
   transparent heuristic so the UI stays usable while you're setting up.

---

## 3. Step-by-step setup

### Step 0 — Prerequisites
- Python 3.10+
- Recommended: a GPU (fine-tuning 3 transformer models on CPU is slow — see
  §6 for a CPU-friendly alternative). Google Colab (free T4 GPU) works well.

### Step 1 — Clone / unzip the project and install dependencies
```bash
cd fake-news-agent
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### Step 2 — Get the dataset
1. Download it from Kaggle:
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Unzip it, then copy `Fake.csv` and `True.csv` into `data/raw/`.

> Kaggle CLI shortcut (if you have `~/.kaggle/kaggle.json` set up):
> ```bash
> kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
> ```

### Step 3 — Preprocess the data
```bash
python src/data_preprocessing.py
```
This cleans the text (strips URLs, wire-service datelines like
`"WASHINGTON (Reuters) -"` which are a trivial shortcut, dedupes), merges
title + body, and writes stratified `train.csv` / `val.csv` / `test.csv`
into `data/processed/`.

### Step 4 — Fine-tune each model
Run once per model (each saves its own checkpoint under `models/<key>/`):
```bash
python src/train.py --model bert
python src/train.py --model roberta
python src/train.py --model deberta
```
Useful flags: `--epochs`, `--batch_size`, `--lr`, `--max_len`
(defaults live in `config.py`). 3 epochs on a T4 GPU takes roughly
15–25 minutes per model on the full ~44k-row dataset.

Or just run everything in one shot:
```bash
bash run_pipeline.sh
```

### Step 5 — Evaluate
```bash
python src/evaluate.py --model all
```
Prints precision/recall/F1/confusion matrix per model and saves a report
to `models/<key>/test_report.txt`.

### Step 6 — Launch the Streamlit app
```bash
streamlit run app.py
```
Open the printed local URL. Paste an article (or upload a `.txt` file),
hit **Run Verification**, and the agent returns a stamped verdict, a
per-model confidence breakdown, and its reasoning.

> The app works even before you've trained any models — it falls back to a
> transparent heuristic so you can see the UI immediately, and will start
> using real predictions as soon as a `models/<key>/` folder exists.

### (Optional) Step 7 — Enable live fact cross-referencing
```bash
export NEWSAPI_KEY="your_newsapi_org_key"   # free tier at newsapi.org
streamlit run app.py
```
Without a key, the app simply skips this section — no crash, no extra config.

---

## 4. How the ensemble verdict is computed

For an article, each model outputs `P(real)` and `P(fake)`. The agent
computes a weighted average of `P(real)` across all loaded models
(weights adjustable live in the Streamlit sidebar, default equal), then:

```
final_label      = "REAL" if weighted_P(real) >= 0.5 else "FAKE"
final_confidence = max(weighted_P(real), 1 - weighted_P(real))
```

It also reports whether the three models agreed, which sentences carried
the most signal, and which sensational phrases were detected.

---

## 5. Extending the project
- **Add a 4th model**: add an entry to `MODEL_REGISTRY` in `config.py` —
  everything else (`train.py`, `evaluate.py`, `agent.py`, `app.py`) picks
  it up automatically.
- **Better claim extraction**: swap `_top_sentences` in `src/agent.py` for
  a proper NER/claim-extraction model.
- **Real fact verification**: extend `cross_reference` to hit a dedicated
  fact-check API (e.g. Google Fact Check Tools API) instead of/alongside
  NewsAPI headlines.
- **Deploy**: `streamlit run app.py` works as-is on Streamlit Community
  Cloud — just make sure trained model weights are available (e.g. via a
  Hugging Face Hub model repo you load in `agent.py`, since checkpoints
  are usually too large to commit to git).

---

## 6. No GPU? Two options
1. **Train on Google Colab** (free T4 GPU): upload the `src/`, `config.py`,
   and `data/raw/*.csv`, run Steps 3–5 there, then download the resulting
   `models/` folder back into this project before running the Streamlit app
   locally.
2. **Use smaller/distilled checkpoints**: swap `hf_name` in `config.py` for
   `distilbert-base-uncased`, `distilroberta-base`, etc. — trains much
   faster on CPU with a small accuracy trade-off.

---

## 7. Disclaimer
This is an educational project. Verdicts are statistical model outputs,
not a substitute for professional fact-checking or journalistic
verification.
