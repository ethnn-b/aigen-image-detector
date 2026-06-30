"""The failure report. This is the deliverable.

It loads a checkpoint, scores the CIFAKE test split, and writes the four things the
usual demo skips:

1. The headline numbers: test accuracy, ROC-AUC, and the operating-point FPR (how
   many real photos get flagged when the detector catches operating_tpr of the fakes).
2. The worst false positives: real images the detector was most sure were fake, copied
   out so they can be looked at.
3. The cross-generator table: the detector's AUC on generators it never trained on.
   Needs GenImage on disk; prints how to get it and skips if it is absent.
4. The degradation curve: AUC as the test images are re-saved at lower JPEG quality and
   shrunk and re-enlarged.

All the scoring goes through metrics.py, which is unit tested. The operating-point
threshold is read from the checkpoint (chosen on validation at train time), so nothing
here is fit on the test set.
"""

from __future__ import annotations

import argparse
import functools
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import (
    ImageItem,
    build_transforms,
    degrade_jpeg,
    degrade_resize,
    load_cifake,
)
from .engine import predict_scores
from .metrics import (
    accuracy,
    cross_generator_drop,
    false_positive_indices,
    fpr_at_tpr,
    roc_auc,
)
from .model import build_model
from .util import get_device

JPEG_QUALITIES = (90, 70, 50, 30, 10)
RESIZE_FACTORS = (0.75, 0.5, 0.25)


def load_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], ckpt["mode"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def _balanced_subset(items: list[ImageItem], n: int, seed: int = 0) -> list[ImageItem]:
    """A class-balanced subset of at most n items, for the cheaper sweeps."""
    rng = np.random.default_rng(seed)
    reals = [it for it in items if it.label == 0]
    fakes = [it for it in items if it.label == 1]
    half = n // 2

    def pick(group: list[ImageItem]) -> list[ImageItem]:
        return [group[i] for i in rng.permutation(len(group))[: min(half, len(group))]]

    return pick(reals) + pick(fakes)


def degradation_curve(model, items, transform, device, batch_size, lines: list[str]) -> None:
    """AUC under JPEG re-encoding and downscale-then-upscale, vs the clean baseline."""
    start = len(lines)
    rows: list[tuple[str, float]] = []
    clean_scores, clean_labels = predict_scores(model, items, transform, device, batch_size)
    rows.append(("clean", roc_auc(clean_scores, clean_labels)))
    for q in JPEG_QUALITIES:
        s, y = predict_scores(
            model, items, transform, device, batch_size, degrade=functools.partial(degrade_jpeg, quality=q)
        )
        rows.append((f"jpeg q={q}", roc_auc(s, y)))
    for f in RESIZE_FACTORS:
        s, y = predict_scores(
            model, items, transform, device, batch_size, degrade=functools.partial(degrade_resize, factor=f)
        )
        rows.append((f"resize x{f}", roc_auc(s, y)))

    lines.append("\n## Degradation curve\n")
    lines.append(f"AUC as the {len(items)} sampled test images are corrupted. A cliff means the")
    lines.append("detector leans on high-frequency content that compression destroys.\n")
    lines.append("| condition | ROC-AUC |")
    lines.append("| --------- | ------- |")
    for name, auc in rows:
        lines.append(f"| {name} | {auc:.4f} |")
    print("\n".join(lines[start:]))


def false_positive_report(scores, labels, items, threshold, k, out_dir: Path, lines: list[str]) -> None:
    start = len(lines)
    idx = false_positive_indices(scores, labels, threshold)
    lines.append("\n## Worst false positives\n")
    lines.append(
        f"Real images scored at or above the operating threshold ({threshold:.4f}), worst first. "
        f"{len(idx)} real images were flagged in all. Top {min(k, len(idx))} copied to "
        f"`{out_dir}` for inspection.\n"
    )
    if not idx:
        lines.append("None: no real image crossed the threshold.")
        print("\n".join(lines[start:]))
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    lines.append("| rank | score | source image |")
    lines.append("| ---- | ----- | ------------ |")
    for rank, i in enumerate(idx[:k], start=1):
        src = Path(items[i].path)
        dst = out_dir / f"fp_{rank:02d}_score{scores[i]:.3f}_{src.name}"
        try:
            shutil.copy(src, dst)
        except OSError:
            dst = src
        lines.append(f"| {rank} | {scores[i]:.4f} | {src.name} |")
    print("\n".join(lines[start:]))


def _load_generator_folder(folder: Path, generator: str) -> list[ImageItem]:
    """Read a GenImage-style generator folder into ImageItems.

    Looks for fake images under any subdir named ai/fake/sd/... and real images under
    nature/real/.... Flexible because GenImage's per-generator layout varies a little.
    """
    fake_names = {"ai", "fake", "fakes", "generated", "sd", "synthetic"}
    real_names = {"nature", "real", "reals", "natural", "imagenet"}
    exts = (".png", ".jpg", ".jpeg", ".webp")
    items: list[ImageItem] = []
    for sub in folder.rglob("*"):
        if not sub.is_file() or sub.suffix.lower() not in exts:
            continue
        parts = {p.lower() for p in sub.parts}
        if parts & fake_names:
            items.append(ImageItem(path=sub, label=1, generator=generator))
        elif parts & real_names:
            items.append(ImageItem(path=sub, label=0, generator="real"))
    return items


def cross_generator_table(
    model, transform, device, batch_size, data_root, same_gen_auc: float, lines: list[str]
) -> None:
    start = len(lines)
    root = Path(data_root) / "genimage"
    lines.append("\n## Cross-generator table\n")
    gen_dirs = sorted([d for d in root.glob("*") if d.is_dir()]) if root.exists() else []
    if not gen_dirs:
        lines.append(
            "Skipped: no GenImage data found under `data/genimage/`. CIFAKE has a single "
            "generator (Stable Diffusion), so the cross-generator drop, the honest headline of "
            "this project, needs GenImage's eight generators. Get it from "
            "https://github.com/GenImage-Dataset/GenImage and drop per-generator folders under "
            "`data/genimage/<generator>/`, then re-run."
        )
        print("\n".join(lines[start:]))
        return

    lines.append(
        "AUC of the CIFAKE-trained detector on generators it never saw in training. The drop "
        f"from the same-generator AUC ({same_gen_auc:.4f}) is the generalization gap.\n"
    )
    lines.append("| test generator | images | ROC-AUC |")
    lines.append("| -------------- | ------ | ------- |")
    cross: dict[str, float] = {}
    for d in gen_dirs:
        items = _load_generator_folder(d, d.name)
        if sum(it.label == 1 for it in items) == 0 or sum(it.label == 0 for it in items) == 0:
            continue
        s, y = predict_scores(model, items, transform, device, batch_size)
        auc = roc_auc(s, y)
        cross[d.name] = auc
        lines.append(f"| {d.name} | {len(items)} | {auc:.4f} |")
    if cross:
        summary = cross_generator_drop(same_gen_auc, cross)
        lines.append(
            f"\nMean cross-generator AUC {summary['mean_cross_gen_auc']:.4f}, "
            f"a drop of {summary['drop']:.4f} from same-generator."
        )
    print("\n".join(lines[start:]))


def evaluate(checkpoint: str, report_path: str = "reports/report.md", fp_examples: int = 12) -> None:
    device = get_device()
    model, ckpt = load_model(checkpoint, device)
    cfg_image_size = ckpt["image_size"]
    threshold = ckpt["val_threshold"]
    operating_tpr = ckpt["operating_tpr"]
    transform = build_transforms(cfg_image_size, train=False)

    splits = load_cifake("data", val_fraction=0.1, seed=0, limit_per_split=None)
    print(f"scoring {len(splits.test)} test images on {device} ...")
    scores, labels = predict_scores(model, splits.test, transform, device, 64)

    acc_half = accuracy((scores >= 0.5).astype(int), labels)
    acc_op = accuracy((scores >= threshold).astype(int), labels)
    auc = roc_auc(scores, labels)
    fpr = fpr_at_tpr(scores, labels, operating_tpr)

    lines: list[str] = []
    lines.append("# Failure report")
    lines.append(
        f"\nCheckpoint `{checkpoint}` ({ckpt['mode']}, backbone `{ckpt['backbone']}`, "
        f"val AUC {ckpt['val_auc']:.4f}). Test set: {len(splits.test)} CIFAKE images.\n"
    )
    lines.append("## Headline\n")
    lines.append("| metric | value |")
    lines.append("| ------ | ----- |")
    lines.append(f"| ROC-AUC | {auc:.4f} |")
    lines.append(f"| accuracy @ 0.5 | {acc_half:.4f} |")
    lines.append(f"| accuracy @ operating threshold | {acc_op:.4f} |")
    lines.append(f"| FPR @ TPR={operating_tpr} | {fpr:.4f} |")
    lines.append(
        f"\nRead the last row as: to catch {operating_tpr:.0%} of the fakes, the detector wrongly "
        f"flags {fpr:.1%} of real photos as fake."
    )
    print("\n".join(lines))

    report = Path(report_path)
    false_positive_report(
        scores, labels, splits.test, threshold, fp_examples, report.parent / "false_positives", lines
    )

    deg_subset = _balanced_subset(splits.test, 4000, seed=0)
    degradation_curve(model, deg_subset, transform, device, 64, lines)

    cross_generator_table(model, transform, device, 64, "data", auc, lines)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(f"\nfull report written to {report}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the failure report")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--report", default="reports/report.md")
    parser.add_argument("--fp-examples", type=int, default=12)
    args = parser.parse_args(argv)
    evaluate(args.checkpoint, args.report, args.fp_examples)


if __name__ == "__main__":
    main()
