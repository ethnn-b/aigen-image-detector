"""Single-image prediction, with an optional Grad-CAM heatmap.

Loads a checkpoint, scores one image as real or fake, and (with --heatmap) saves an
overlay showing where the model looked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import build_transforms
from .explain import gradcam, overlay_heatmap
from .model import build_model
from .util import get_device


def load_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], ckpt["mode"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def predict(image_path: str, checkpoint: str, heatmap: bool = False) -> dict:
    """Return {label, prob_fake} for one image; optionally save a heatmap overlay.

    prob_fake is the softmax probability of the fake class. The label is read against
    the operating-point threshold stored in the checkpoint, the same threshold the
    failure report uses, so the demo and the report agree on what counts as fake.
    """
    from PIL import Image

    device = get_device()
    model, ckpt = load_model(checkpoint, device)
    transform = build_transforms(ckpt["image_size"], train=False)
    threshold = ckpt.get("val_threshold", 0.5)

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_fake = float(torch.softmax(model(x), dim=1)[0, 1])
    label = "fake" if prob_fake >= threshold else "real"

    result = {"label": label, "prob_fake": prob_fake, "threshold": threshold}
    print(f"{image_path}: {label} (P(fake)={prob_fake:.4f}, threshold={threshold:.4f})")

    if heatmap:
        cam = gradcam(model, transform(img), target_class=1)
        out = Path(image_path).with_suffix("").as_posix() + "_heatmap.png"
        overlay_heatmap(img, cam).save(out)
        result["heatmap_path"] = out
        print(f"heatmap saved to {out}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classify one image as real or AI-generated")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--heatmap", action="store_true", help="also save a Grad-CAM heatmap")
    args = parser.parse_args(argv)
    predict(args.image, args.checkpoint, args.heatmap)


if __name__ == "__main__":
    main()
