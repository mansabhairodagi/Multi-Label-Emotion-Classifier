"""
Train the multi-label emotion ensemble and save artifacts to ./artifacts/

Usage (from the project root):
  python -m src.main
  python -m src.main --data data/track-a.csv --out artifacts
"""

import os
import sys
from pathlib import Path

# Must be set before importing TensorFlow / Keras
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk

sys.modules["keras"] = tfk

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from src.emotion_model import (
    ARTIFACT_DIR,
    DEFAULT_TRAIN_CSV,
    build_bert,
    save_artifacts,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise"]
HF_MODEL = "distilbert-base-uncased"
MAX_NUM_WORDS = 10_000
MAX_SEQ_LEN = 100


def cluster_based_split(X, y, *, test_size=0.2, val_size=0.2, random_state=None):
    """Cluster-aware train/val/test split for multi-label data."""
    rng = np.random.RandomState(random_state)

    kmeans = KMeans(n_clusters=5, random_state=random_state)
    clusters = kmeans.fit_predict(y)
    uniq = np.unique(clusters)

    train_val_idx, test_idx = [], []
    for c in uniq:
        idx_c = np.where(clusters == c)[0].copy()
        rng.shuffle(idx_c)
        split = int(len(idx_c) * (1 - test_size))
        train_val_idx.extend(idx_c[:split])
        test_idx.extend(idx_c[split:])

    clusters_tv = clusters[train_val_idx]
    train_idx, val_idx = [], []
    for c in uniq:
        idx_c = np.where(clusters_tv == c)[0].copy()
        rng.shuffle(idx_c)
        split = int(len(idx_c) * (1 - val_size / (1 - test_size)))
        train_idx.extend(idx_c[:split])
        val_idx.extend(idx_c[split:])

    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    return (
        X_tv[train_idx],
        X_tv[val_idx],
        X[test_idx],
        y_tv[train_idx],
        y_tv[val_idx],
        y[test_idx],
    )


def report_f1(name, y_true, probs_or_pred, thr=None):
    if thr is not None:
        y_hat = (probs_or_pred >= thr).astype(int)
    else:
        y_hat = probs_or_pred
    score = f1_score(y_true, y_hat, average="macro", zero_division=0)
    print(f"{name:12s} Macro-F1: {score:.3f}")
    return score


def train(
    data_path: str = str(DEFAULT_TRAIN_CSV),
    artifact_dir: str = str(ARTIFACT_DIR),
):
    df = pd.read_csv(data_path).dropna(subset=["text"])
    emotion_labels = EMOTION_LABELS

    X_text = df["text"].to_numpy()
    Y = df[emotion_labels].to_numpy(dtype="float32")

    perm = np.random.permutation(len(X_text))
    X_text, Y = X_text[perm], Y[perm]

    X_train, X_val, X_test, y_train, y_val, y_test = cluster_based_split(
        X_text, Y, test_size=0.20, val_size=0.20, random_state=SEED
    )
    X_train_text, X_val_text, X_test_text = map(list, [X_train, X_val, X_test])
    print(
        f"Split sizes — train: {len(X_train_text)}, "
        f"val: {len(X_val_text)}, test: {len(X_test_text)}"
    )

    # --- TF-IDF + LR --------------------------------------------------------
    print("\n=== Training TF-IDF + Logistic Regression ===")
    tfidf = TfidfVectorizer(
        max_features=20_000, ngram_range=(1, 2), stop_words="english"
    )
    X_tr_tfidf = tfidf.fit_transform(X_train_text)
    X_va_tfidf = tfidf.transform(X_val_text)
    X_te_tfidf = tfidf.transform(X_test_text)

    logreg = OneVsRestClassifier(
        LogisticRegression(max_iter=400, class_weight="balanced")
    )
    logreg.fit(X_tr_tfidf, y_train)

    val_prob, test_prob = {}, {}
    val_prob["tfidf"] = logreg.predict_proba(X_va_tfidf)
    test_prob["tfidf"] = logreg.predict_proba(X_te_tfidf)
    report_f1("TF-IDF+LR", y_test, test_prob["tfidf"], thr=0.5)

    # --- Bi-LSTM ------------------------------------------------------------
    print("\n=== Training Bi-LSTM + Attention ===")
    tok = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(X_train_text)

    def to_seq(txt_iterable):
        return pad_sequences(
            tok.texts_to_sequences(txt_iterable), maxlen=MAX_SEQ_LEN
        )

    X_tr_seq, X_va_seq, X_te_seq = map(
        to_seq, [X_train_text, X_val_text, X_test_text]
    )

    pos = np.sum(y_train, axis=0).astype(np.float32)
    neg = float(len(y_train)) - pos
    pos_w_tf = tf.constant(neg / (pos + 1e-6), dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        sample_w = y_true * pos_w_tf + (1.0 - y_true)
        return tf.reduce_mean(bce * sample_w)

    inp = layers.Input(shape=(MAX_SEQ_LEN,))
    x = layers.Embedding(MAX_NUM_WORDS, 128)(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Attention()([x, x])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(emotion_labels), activation="sigmoid")(x)

    bilstm = tfk.Model(inp, out)
    bilstm.compile(
        optimizer=tfk.optimizers.legacy.Adam(1e-3),
        loss=weighted_bce,
    )
    bilstm.fit(
        X_tr_seq,
        y_train,
        validation_data=(X_va_seq, y_val),
        epochs=15,
        batch_size=64,
        verbose=1,
        callbacks=[
            tfk.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        ],
    )

    val_prob["bilstm"] = bilstm.predict(X_va_seq, batch_size=256, verbose=0)
    test_prob["bilstm"] = bilstm.predict(X_te_seq, batch_size=256, verbose=0)
    report_f1("Bi-LSTM", y_test, test_prob["bilstm"], thr=0.5)

    # --- DistilBERT ---------------------------------------------------------
    print("\n=== Training DistilBERT ===")
    hf_tok = AutoTokenizer.from_pretrained(HF_MODEL)

    def bert_enc(txt_iterable):
        return hf_tok(
            list(txt_iterable),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="tf",
        )

    tr_enc, va_enc, te_enc = map(
        bert_enc, [X_train_text, X_val_text, X_test_text]
    )

    bert = build_bert(HF_MODEL, len(emotion_labels))
    print(
        "DistilBERT trainable variables:",
        len(bert.trainable_variables),
        "(includes Dense head)",
    )

    bert.compile(
        optimizer=tfk.optimizers.Adam(learning_rate=2e-5),
        loss=weighted_bce,
    )
    bert.fit(
        {
            "input_ids": tr_enc["input_ids"],
            "attention_mask": tr_enc["attention_mask"],
        },
        y_train,
        validation_data=(
            {
                "input_ids": va_enc["input_ids"],
                "attention_mask": va_enc["attention_mask"],
            },
            y_val,
        ),
        epochs=3,
        batch_size=32,
        verbose=1,
    )

    val_prob["bert"] = bert.predict(
        {
            "input_ids": va_enc["input_ids"],
            "attention_mask": va_enc["attention_mask"],
        },
        batch_size=64,
        verbose=0,
    )
    test_prob["bert"] = bert.predict(
        {
            "input_ids": te_enc["input_ids"],
            "attention_mask": te_enc["attention_mask"],
        },
        batch_size=64,
        verbose=0,
    )
    report_f1("DistilBERT", y_test, test_prob["bert"], thr=0.5)

    # --- Ensemble + thresholds ----------------------------------------------
    print("\n=== Ensemble + threshold tuning ===")
    model_names = list(val_prob.keys())
    val_stack = np.stack([val_prob[n] for n in model_names], axis=0)
    test_stack = np.stack([test_prob[n] for n in model_names], axis=0)

    raw_w = []
    for name in model_names:
        w = f1_score(
            y_val,
            (val_prob[name] >= 0.5).astype(int),
            average="macro",
            zero_division=0,
        )
        raw_w.append(max(w, 1e-6))
        print(f"  val Macro-F1[{name}] = {w:.3f}")

    weights = np.asarray(raw_w, dtype=np.float64)
    weights = weights / weights.sum()
    print("  soft-vote weights:", dict(zip(model_names, np.round(weights, 3))))

    val_mean = np.tensordot(weights, val_stack, axes=(0, 0))
    test_mean = np.tensordot(weights, test_stack, axes=(0, 0))

    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr = np.zeros(len(emotion_labels))
    for j, lbl in enumerate(emotion_labels):
        best_f1, best_t = 0.0, 0.5
        for t in thr_grid:
            f1 = f1_score(
                y_val[:, j],
                (val_mean[:, j] >= t).astype(int),
                zero_division=0,
            )
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thr[j] = best_t
        print(f"  threshold[{lbl}] = {best_t:.2f} (val F1={best_f1:.3f})")

    y_pred = (test_mean >= best_thr).astype(int)
    print("\n=== Ensemble hold-out results ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=emotion_labels,
            digits=3,
            zero_division=0,
        )
    )
    ensemble_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print("Ensemble Macro-F1 :", ensemble_f1)

    metrics = {
        "ensemble_macro_f1": float(ensemble_f1),
        "per_model_test_macro_f1_at_0.5": {
            name: float(
                f1_score(
                    y_test,
                    (test_prob[name] >= 0.5).astype(int),
                    average="macro",
                    zero_division=0,
                )
            )
            for name in model_names
        },
    }

    out = save_artifacts(
        artifact_dir=artifact_dir,
        tfidf=tfidf,
        logreg=logreg,
        lstm_tokenizer=tok,
        bilstm=bilstm,
        bert=bert,
        hf_tokenizer=hf_tok,
        emotion_labels=emotion_labels,
        model_names=model_names,
        weights=weights,
        best_thr=best_thr,
        hf_model=HF_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        max_num_words=MAX_NUM_WORDS,
        metrics=metrics,
    )
    print(f"\nSaved trained models to {out.resolve()}")
    print("Predict without retraining:  python -m src.predict <csv_or_text>")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train emotion ensemble and save artifacts"
    )
    ap.add_argument(
        "--data",
        default=str(DEFAULT_TRAIN_CSV),
        help="Training CSV path (default: data/track-a.csv)",
    )
    ap.add_argument(
        "--out",
        default=str(ARTIFACT_DIR),
        help="Artifact directory (default: artifacts)",
    )
    args = ap.parse_args()
    train(data_path=args.data, artifact_dir=args.out)
