"""
Predict emotions from a saved ensemble (no retraining).

Usage (from the project root):
  python -m src.predict data/track-a-test.csv
  python -m src.predict --text "I am so happy I could cry"
  python -m src.predict data/track-a-test.csv --out my_preds.csv --artifacts artifacts
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk

sys.modules["keras"] = tfk

from src.emotion_model import ARTIFACT_DIR, EmotionEnsemble


def main():
    ap = argparse.ArgumentParser(
        description="Emotion ensemble prediction (loads ./artifacts)"
    )
    ap.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Input CSV with a 'text' column",
    )
    ap.add_argument(
        "--text",
        "-t",
        default=None,
        help="Classify a single text string instead of a CSV",
    )
    ap.add_argument(
        "--artifacts",
        default=str(ARTIFACT_DIR),
        help="Directory with saved models (default: artifacts)",
    )
    ap.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output CSV path (default: <input>_predicted.csv)",
    )
    args = ap.parse_args()

    if not args.csv and not args.text:
        ap.error("Provide a CSV path or --text")

    print(f"Loading models from {args.artifacts} ...")
    ensemble = EmotionEnsemble(args.artifacts)
    labels = ensemble.emotion_labels
    print(f"Ready. Labels: {labels}")

    if args.text is not None:
        probs, label_lists = ensemble.predict(args.text)
        print("\nText:", args.text)
        print("Predicted:", label_lists[0] if label_lists[0] else "(none)")
        print("Probabilities:")
        for name, p in zip(labels, probs[0]):
            print(f"  {name:10s} {p:.3f}  thr={ensemble.best_thr[labels.index(name)]:.2f}")
        return

    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise ValueError("Expected the CSV to contain a 'text' column.")

    probs, label_lists = ensemble.predict(df["text"].astype(str).tolist())
    df["predicted_labels"] = label_lists
    for i, name in enumerate(labels):
        df[f"prob_{name}"] = probs[:, i]

    out_path = args.out or (
        os.path.splitext(args.csv)[0] + "_predicted.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    print(df[["text", "predicted_labels"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
