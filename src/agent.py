"""
The "agent" layer.

This is what makes the project more than three separate classifiers:
FakeNewsAgent orchestrates BERT / RoBERTa / DeBERTa, combines their
votes into a single verdict + confidence, extracts the sentences that
most influenced the decision, and (optionally, if NEWSAPI_KEY is set)
cross-references the article's core claim against live headlines for
a lightweight fact-verification signal.

Everything degrades gracefully: if a model wasn't trained yet, it's
skipped and the ensemble reweights over whatever is available. If no
model is trained, the agent still works using heuristic scoring so the
Streamlit UI never hard-crashes.
"""
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "you won't believe", "secret", "exposed",
    "miracle", "banned", "conspiracy", "they don't want you to know",
    "breaking", "urgent", "outrage", "destroyed", "slams", "bombshell",
}


@dataclass
class ModelVerdict:
    model_key: str
    display_name: str
    label: str
    confidence: float
    prob_real: float
    prob_fake: float


@dataclass
class AgentVerdict:
    final_label: str
    final_confidence: float
    per_model: list = field(default_factory=list)
    flagged_phrases: list = field(default_factory=list)
    key_sentences: list = field(default_factory=list)
    explanation: str = ""
    models_used: int = 0


class FakeNewsAgent:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self._load_available_models()

    def _load_available_models(self):
        for key, info in config.MODEL_REGISTRY.items():
            save_dir = info["save_dir"]
            if os.path.isdir(save_dir) and os.listdir(save_dir):
                try:
                    tok = AutoTokenizer.from_pretrained(save_dir)
                    mdl = AutoModelForSequenceClassification.from_pretrained(save_dir).to(self.device)
                    mdl.eval()
                    self.models[key] = mdl
                    self.tokenizers[key] = tok
                except Exception as e:  # noqa: BLE001
                    print(f"Could not load {key} from {save_dir}: {e}")

    @property
    def available_models(self):
        return list(self.models.keys())

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------
    def _predict_single(self, model_key, text, max_len=256):
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]

        encoding = tokenizer(
            text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = model(**encoding).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_fake, prob_real = float(probs[0]), float(probs[1])
        label = "REAL" if prob_real >= prob_fake else "FAKE"
        confidence = max(prob_real, prob_fake)

        return ModelVerdict(
            model_key=model_key,
            display_name=config.MODEL_REGISTRY[model_key]["display_name"],
            label=label,
            confidence=confidence,
            prob_real=prob_real,
            prob_fake=prob_fake,
        )

    def _heuristic_fallback(self, text):
        """Used only when zero models are trained yet, so the UI is
        still demonstrable. NOT a substitute for the real models."""
        lowered = text.lower()
        hits = sum(1 for phrase in SENSATIONAL_WORDS if phrase in lowered)
        exclaim = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        risk = min(1.0, 0.15 * hits + 0.05 * exclaim + 2.0 * caps_ratio)
        label = "FAKE" if risk > 0.5 else "REAL"
        confidence = 0.5 + abs(risk - 0.5)
        return ModelVerdict(
            model_key="heuristic",
            display_name="Heuristic fallback (no trained models found)",
            label=label,
            confidence=confidence,
            prob_real=1 - risk,
            prob_fake=risk,
        )

    def predict(self, text, weights=None) -> AgentVerdict:
        weights = weights or config.ENSEMBLE_WEIGHTS
        text = text.strip()

        per_model = []
        if not self.models:
            per_model = [self._heuristic_fallback(text)]
        else:
            for key in self.models:
                per_model.append(self._predict_single(key, text))

        # Weighted vote over prob_real
        total_weight = sum(weights.get(v.model_key, 1.0) for v in per_model) or 1.0
        weighted_real = sum(v.prob_real * weights.get(v.model_key, 1.0) for v in per_model) / total_weight
        final_label = "REAL" if weighted_real >= 0.5 else "FAKE"
        final_confidence = max(weighted_real, 1 - weighted_real)

        flagged = self._flag_sensational_phrases(text)
        key_sentences = self._top_sentences(text, final_label)
        explanation = self._build_explanation(final_label, final_confidence, per_model, flagged)

        return AgentVerdict(
            final_label=final_label,
            final_confidence=final_confidence,
            per_model=per_model,
            flagged_phrases=flagged,
            key_sentences=key_sentences,
            explanation=explanation,
            models_used=len(self.models),
        )

    # ------------------------------------------------------------------
    # Explainability helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flag_sensational_phrases(text):
        lowered = text.lower()
        return sorted({phrase for phrase in SENSATIONAL_WORDS if phrase in lowered})

    @staticmethod
    def _top_sentences(text, label, k=3):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 4]
        if not sentences:
            return []
        # Simple salience heuristic: sentences containing sensational
        # words or numbers/quotes tend to carry the most signal.
        def score(s):
            lowered = s.lower()
            sc = sum(1 for w in SENSATIONAL_WORDS if w in lowered)
            sc += len(re.findall(r"\d", s)) * 0.1
            sc += s.count('"') * 0.3
            return sc

        ranked = sorted(sentences, key=score, reverse=True)
        return ranked[:k]

    @staticmethod
    def _build_explanation(label, confidence, per_model, flagged):
        agree = len({v.label for v in per_model}) == 1
        parts = []
        if agree:
            parts.append(
                f"All {len(per_model)} model(s) agree the article is likely {label} "
                f"(ensemble confidence {confidence:.1%})."
            )
        else:
            votes = ", ".join(f"{v.display_name.split(' (')[0]}={v.label}" for v in per_model)
            parts.append(
                f"Models disagree ({votes}); the weighted ensemble leans {label} "
                f"({confidence:.1%})."
            )
        if flagged:
            parts.append(
                f"Detected {len(flagged)} sensationalist phrase(s) often correlated with "
                f"misinformation: {', '.join(flagged)}."
            )
        else:
            parts.append("No strongly sensationalist language patterns detected.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Optional live fact cross-referencing (NewsAPI). Fully optional —
    # skipped silently if no key / no network.
    # ------------------------------------------------------------------
    def cross_reference(self, query, max_results=5):
        if not config.NEWSAPI_KEY:
            return {"enabled": False, "articles": []}
        try:
            import requests

            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query[:200],
                    "sortBy": "relevancy",
                    "pageSize": max_results,
                    "apiKey": config.NEWSAPI_KEY,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = [
                {"title": a["title"], "source": a["source"]["name"], "url": a["url"]}
                for a in data.get("articles", [])
            ]
            return {"enabled": True, "articles": articles}
        except Exception as e:  # noqa: BLE001
            return {"enabled": True, "error": str(e), "articles": []}
