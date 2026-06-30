"""Gradio demo: upload an image, get real-or-fake with a confidence and a heatmap.

Needs the demo extra (uv sync --extra demo) and a trained checkpoint. The model is
loaded once, lazily, so the first prediction pays for it and the rest are quick.

    uv run python -m aidetect.app
    AIDETECT_CKPT_FILE=checkpoints/best_finetune.pt uv run python -m aidetect.app
"""

from __future__ import annotations

import os

import torch

from .data import build_transforms
from .explain import gradcam, overlay_heatmap
from .model import build_model
from .util import get_device

_STATE: dict = {}


def _ensure_model(checkpoint: str):
    if "model" not in _STATE:
        device = get_device()
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_model(ckpt["backbone"], ckpt["mode"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        _STATE.update(
            model=model,
            device=device,
            transform=build_transforms(ckpt["image_size"], train=False),
            threshold=ckpt.get("val_threshold", 0.5),
        )
    return _STATE


def classify(image, checkpoint: str):
    """Return (label dict for gr.Label, heatmap image) for one uploaded image."""
    s = _ensure_model(checkpoint)
    x = s["transform"](image.convert("RGB")).unsqueeze(0).to(s["device"])
    with torch.no_grad():
        prob_fake = float(torch.softmax(s["model"](x), dim=1)[0, 1])
    cam = gradcam(s["model"], s["transform"](image.convert("RGB")), target_class=1)
    overlay = overlay_heatmap(image, cam)
    label = "fake" if prob_fake >= s["threshold"] else "real"
    return {"AI-generated (fake)": prob_fake, "real": 1.0 - prob_fake}, overlay, label


def build_interface(checkpoint: str):
    import gradio as gr

    with gr.Blocks(title="AI-image detector") as demo:
        gr.Markdown(
            "# Real or AI-generated?\n"
            "Upload an image. The detector returns the probability it is AI-generated, the "
            "real-or-fake call at the report's operating threshold, and a Grad-CAM heatmap of "
            "where it looked. Trained on CIFAKE; expect the score to be less reliable on "
            "generators it never saw, which is the whole point of the failure report."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="image")
            with gr.Column():
                out_label = gr.Label(label="scores")
                out_call = gr.Textbox(label="call at operating threshold")
                out_heat = gr.Image(label="Grad-CAM")
        gr.Button("classify").click(
            fn=lambda img: (lambda r: (r[0], r[2], r[1]))(classify(img, checkpoint)),
            inputs=inp,
            outputs=[out_label, out_call, out_heat],
        )
    return demo


def main() -> None:
    checkpoint = os.environ.get("AIDETECT_CKPT_FILE", "checkpoints/best.pt")
    build_interface(checkpoint).launch()


if __name__ == "__main__":
    main()
