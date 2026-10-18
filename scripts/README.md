# Scripts

Scripts here are **optional** and not used by the main train → predict → Gradio path.

| File | Notes |
|------|--------|
| `DistillBERT.py` | Standalone PyTorch DistilBERT multi-label trainer (requires `torch`) |

`DistillBERT.py` reads `data/track-a.csv` from the repository root and writes its checkpoint next to this script.

Use `python -m src.main`, `python -m src.predict`, and `python -m src.app` for the portfolio demo.
