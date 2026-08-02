"""
Gradio demo for the multi-label emotion ensemble.

Usage (from project root, after training):
  python app.py
Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import sys

# TF / Keras must be configured before Gradio pulls in its web stack
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from tensorflow import keras as tfk

sys.modules["keras"] = tfk

from emotion_model import ARTIFACT_DIR, EmotionEnsemble

EXAMPLES = [
    ["I am so happy I could cry with joy!"],
    ["Walking alone in the dark fog made my heart race."],
    ["I can't believe they lied to me — I'm furious."],
    ["Losing that match left me empty and quiet for days."],
    ["Wait… they threw a surprise party for me?"],
]

_ENSEMBLE: EmotionEnsemble | None = None


def get_ensemble() -> EmotionEnsemble:
    """Load artifacts once (after TF is ready; avoids Gradio import-order hangs)."""
    global _ENSEMBLE
    if _ENSEMBLE is None:
        meta = os.path.join(str(ARTIFACT_DIR), "meta.json")
        if not os.path.exists(meta):
            raise FileNotFoundError(
                f"No trained models in '{ARTIFACT_DIR}'. Run: python main.py"
            )
        print(f"Loading ensemble from {ARTIFACT_DIR} ...", flush=True)
        _ENSEMBLE = EmotionEnsemble(ARTIFACT_DIR)
        print("Models ready.", flush=True)
    return _ENSEMBLE


def predict_emotions(text: str):
    ens = get_ensemble()
    labels = ens.emotion_labels
    text = (text or "").strip()
    if not text:
        return "Enter some text to classify.", {name: 0.0 for name in labels}

    probs, label_lists = ens.predict(text)
    row = probs[0]
    thr = ens.best_thr
    predicted = label_lists[0]

    if predicted:
        summary = "### Predicted: " + ", ".join(f"**{p}**" for p in predicted)
    else:
        summary = "### Predicted: *(none above threshold)*"

    lines = [
        "| Emotion | Prob | Threshold | Active |",
        "|---------|------|-----------|--------|",
    ]
    for i, name in enumerate(labels):
        active = "yes" if row[i] >= thr[i] else ""
        lines.append(f"| {name} | {row[i]:.3f} | {thr[i]:.2f} | {active} |")
    summary = summary + "\n\n" + "\n".join(lines)

    confidences = {name: float(row[i]) for i, name in enumerate(labels)}
    return summary, confidences


def build_ui():
    import gradio as gr  # import after TF setup

    with gr.Blocks(title="Emotion Classification") as demo:
        gr.Markdown(
            """
            # Multi-label Emotion Classifier
            Ensemble of **TF-IDF + LR**, **Bi-LSTM**, and **DistilBERT**  
            Emotions: joy · sadness · anger · fear · surprise  
            *(one text can trigger multiple labels)*

            *First prediction loads models (~1–3 min on CPU).*
            """
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(
                    label="Text",
                    placeholder="Type a sentence or short paragraph…",
                    lines=5,
                )
                btn = gr.Button("Classify", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=inp)
            with gr.Column():
                out_md = gr.Markdown()
                out_label = gr.Label(label="Probabilities", num_top_classes=5)

        btn.click(fn=predict_emotions, inputs=inp, outputs=[out_md, out_label])
        inp.submit(fn=predict_emotions, inputs=inp, outputs=[out_md, out_label])
        gr.Markdown(
            "_Loads local `artifacts/`. Train first with `python main.py` if needed._"
        )
    return demo


if __name__ == "__main__":
    print("Starting Gradio UI (models load on first Classify)...", flush=True)
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True,
    )
