# Architecture

How the training and inference paths fit together, and why the main design choices exist.

## End-to-end flow

```text
data/track-a.csv
    │
    ▼
cluster_based_split (K-Means on multi-hot labels)
    │
    ├── TF-IDF + OvR LogisticRegression  ──► probs
    ├── Bi-LSTM + Attention (Keras)      ──► probs
    └── DistilBERT (TF, subclassed head) ──► probs
                │
                ▼
     validation Macro-F1 → soft-vote weights
                │
     per-emotion F1 threshold grid search (val)
                │
                ▼
         hold-out Macro-F1 report
                │
                ▼
         artifacts/  (models, tokenizers, meta.json)
                │
        ┌───────┴────────┐
        ▼                ▼
   src/predict.py     src/app.py (Gradio)
```

## Module roles

| File | Responsibility |
|------|----------------|
| `src/main.py` | Train all models, tune the ensemble, evaluate, write `artifacts/` |
| `src/emotion_model.py` | `DistilBertMultiLabel`, `save_artifacts()`, `EmotionEnsemble` loader |
| `src/predict.py` | CLI inference only (no training) |
| `src/app.py` | Gradio UI; loads `EmotionEnsemble` on first classify |
| `data/` | Training CSV and sample inference CSV |
| `dashboards/` | Optional Power BI report |
| `docs/images/` | Screenshots used by the README |
| `experiments/DistillBERT.py` | Standalone PyTorch trainer; not used by the main path |

## Design notes

### Multi-label outputs
Labels are independent binary targets. Models emit five probabilities; a text can activate several emotions or none.

### Cluster-aware split
K-Means on the label matrix groups similar co-occurrence patterns before slicing train / val / test, which is useful on a small, imbalanced multi-label set.

### Weighted soft voting
Equal averaging let the weaker Bi-LSTM pull down DistilBERT. Weights proportional to validation Macro-F1 reduce that effect.

### Per-label thresholds
Under class imbalance, a single 0.5 cutoff is a poor fit. Each emotion’s threshold is chosen by grid search on validation F1.

### Subclassed DistilBERT head
Wrapping `TFAutoModel` in the Functional API left classification Dense layers poorly tracked. A subclassed `tf.keras.Model` trains the head correctly.

### Saved artifacts
`artifacts/meta.json` stores emotion names, soft-vote weights, thresholds, and reported metrics so `src/predict.py` and `src/app.py` do not retrain.

## Scope

- DistilBERT is the strongest single model; the Bi-LSTM is weaker and down-weighted in the ensemble.
- Power BI is optional batch analytics; the Gradio app is the interactive interface.
- `experiments/DistillBERT.py` is an alternate implementation, not the default pipeline.
