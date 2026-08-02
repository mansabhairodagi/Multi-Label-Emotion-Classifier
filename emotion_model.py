"""Shared DistilBERT model + ensemble save/load helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Prefer legacy Keras when this module is imported first
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModel

ARTIFACT_DIR = Path("artifacts")
HF_MAX_LEN = 128


class DistilBertMultiLabel(tfk.Model):
    """DistilBERT + classification head (subclassed for proper weight tracking)."""

    def __init__(self, model_name: str, n_labels: int, **kwargs):
        super().__init__(**kwargs)
        self.transformer = TFAutoModel.from_pretrained(model_name)
        self.dense = layers.Dense(256, activation="relu")
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(n_labels, activation="sigmoid")

    def call(self, inputs, training=False):
        if isinstance(inputs, (list, tuple)):
            input_ids, attention_mask = inputs
        else:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        seq = self.transformer(
            input_ids, attention_mask=attention_mask, training=training
        )[0]
        x = self.dense(seq[:, 0, :])
        x = self.dropout(x, training=training)
        return self.classifier(x)


def build_bert(model_name: str, n_labels: int) -> DistilBertMultiLabel:
    """Create and build the DistilBERT model so weights can be loaded."""
    model = DistilBertMultiLabel(model_name, n_labels)
    dummy_ids = tf.zeros((1, HF_MAX_LEN), dtype=tf.int32)
    dummy_mask = tf.ones((1, HF_MAX_LEN), dtype=tf.int32)
    _ = model(
        {"input_ids": dummy_ids, "attention_mask": dummy_mask},
        training=False,
    )
    return model


def save_artifacts(
    *,
    artifact_dir: str | Path,
    tfidf,
    logreg,
    lstm_tokenizer,
    bilstm,
    bert: DistilBertMultiLabel,
    hf_tokenizer,
    emotion_labels: list[str],
    model_names: list[str],
    weights: np.ndarray,
    best_thr: np.ndarray,
    hf_model: str,
    max_seq_len: int,
    max_num_words: int,
    metrics: dict | None = None,
) -> Path:
    """Persist all ensemble components for offline prediction."""
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(tfidf, out / "tfidf.joblib")
    joblib.dump(logreg, out / "logreg.joblib")
    joblib.dump(lstm_tokenizer, out / "lstm_tokenizer.joblib")

    bilstm.save(out / "bilstm.keras")
    bert.save_weights(str(out / "bert.weights.h5"))
    hf_tokenizer.save_pretrained(str(out / "hf_tokenizer"))

    meta = {
        "emotion_labels": list(emotion_labels),
        "model_names": list(model_names),
        "weights": [float(w) for w in weights],
        "best_thr": [float(t) for t in best_thr],
        "hf_model": hf_model,
        "max_seq_len": int(max_seq_len),
        "max_num_words": int(max_num_words),
        "hf_max_len": HF_MAX_LEN,
    }
    if metrics:
        meta["metrics"] = metrics

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out


class EmotionEnsemble:
    """Load saved artifacts and run multi-label ensemble inference."""

    def __init__(self, artifact_dir: str | Path = ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        meta_path = self.artifact_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained artifacts at {self.artifact_dir.resolve()}. "
                f"Run: python main.py"
            )

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        self.emotion_labels = self.meta["emotion_labels"]
        self.model_names = self.meta["model_names"]
        self.weights = np.asarray(self.meta["weights"], dtype=np.float64)
        self.best_thr = np.asarray(self.meta["best_thr"], dtype=np.float64)
        self.max_seq_len = int(self.meta["max_seq_len"])
        self.hf_max_len = int(self.meta.get("hf_max_len", HF_MAX_LEN))

        self.tfidf = joblib.load(self.artifact_dir / "tfidf.joblib")
        self.logreg = joblib.load(self.artifact_dir / "logreg.joblib")
        self.lstm_tokenizer = joblib.load(
            self.artifact_dir / "lstm_tokenizer.joblib"
        )
        self.bilstm = tfk.models.load_model(
            self.artifact_dir / "bilstm.keras",
            compile=False,
        )
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            str(self.artifact_dir / "hf_tokenizer")
        )
        self.bert = build_bert(
            self.meta["hf_model"], len(self.emotion_labels)
        )
        weights_path = str(self.artifact_dir / "bert.weights.h5")
        print(f"Loading DistilBERT weights from {weights_path} ...", flush=True)
        self.bert.load_weights(weights_path)
        print("DistilBERT weights loaded.", flush=True)

    def _prep_seq(self, texts: list[str]) -> np.ndarray:
        return pad_sequences(
            self.lstm_tokenizer.texts_to_sequences(texts),
            maxlen=self.max_seq_len,
        )

    def _prep_bert(self, texts: list[str]) -> dict:
        enc = self.hf_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.hf_max_len,
            return_tensors="tf",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    def predict(self, texts: str | list[str]):
        """
        Returns
        -------
        probs : np.ndarray, shape (n_samples, n_labels)
        label_lists : list[list[str]]
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]

        probs_by_model = {
            "tfidf": self.logreg.predict_proba(self.tfidf.transform(texts)),
            "bilstm": self.bilstm.predict(
                self._prep_seq(texts), batch_size=256, verbose=0
            ),
            "bert": self.bert.predict(
                self._prep_bert(texts), batch_size=64, verbose=0
            ),
        }

        p_mean = np.zeros_like(probs_by_model[self.model_names[0]], dtype=np.float64)
        for name, w in zip(self.model_names, self.weights):
            p_mean += w * probs_by_model[name]

        y_hat = p_mean >= self.best_thr
        label_lists = [
            [self.emotion_labels[j] for j, flag in enumerate(row) if flag]
            for row in y_hat
        ]
        return p_mean, label_lists
