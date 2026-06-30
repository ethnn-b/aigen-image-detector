"""The cross-generator experiment, the project's headline result.

CIFAKE has a single generator, so it cannot show cross-generator generalization on
its own. This pulls small, balanced slices of GenImage (several generators) plus a
shared real set from the Hugging Face hub, keyless, then runs the honest test:

    train a linear probe on ONE generator (MidJourney) with a shared real
    distribution, then measure its AUC on generators it never saw (BigGAN, ADM,
    glide). The drop from the same-generator AUC is the generalization gap.

The real distribution is shared across train and every test set, and the held-out
generators appear in no training image, so the gap is generator shift and not domain
shift. Images come through the datasets-server /rows endpoint (individual URLs), so
this downloads a few hundred images per source, not the full multi-GB parquet shards.

    uv run python scripts/cross_generator_experiment.py
    uv run python scripts/cross_generator_experiment.py --per-class 400
"""

from __future__ import annotations

import argparse
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from aidetect.data import ImageItem, build_transforms
from aidetect.engine import extract_all_features
from aidetect.metrics import cross_generator_drop, fpr_at_tpr, roc_auc
from aidetect.model import build_model, classifier_module
from aidetect.train import fit_head
from aidetect.util import get_device, seed_everything

REAL_SOURCE = "bitmind/bm-real"
GEN_SOURCES = {
    "midjourney": "bitmind/GenImage_MidJourney",
    "biggan": "bitmind/GenImage_BigGAN",
    "adm": "bitmind/GenImage_ADM",
    "glide": "bitmind/GenImage_glide",
}
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def fetch_urls(dataset: str, n: int) -> list[str]:
    """Page the datasets-server /rows endpoint for up to n image URLs."""
    urls: list[str] = []
    offset = 0
    while len(urls) < n:
        length = min(100, n - len(urls))
        params = {"dataset": dataset, "config": "default", "split": "train",
                  "offset": offset, "length": length}
        r = requests.get(ROWS_URL, params=params, timeout=60).json()
        rows = r.get("rows", [])
        if not rows:
            break
        for row in rows:
            cell = row["row"].get("image")
            if isinstance(cell, dict) and cell.get("src"):
                urls.append(cell["src"])
        offset += length
    return urls[:n]


def _download_one(task: tuple[str, Path]) -> bool:
    url, out = task
    if out.exists():
        return True
    try:
        data = requests.get(url, timeout=60).content
        Image.open(io.BytesIO(data)).convert("RGB").save(out, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def download_source(name: str, dataset: str, n: int, root: Path) -> list[Path]:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(d.glob("*.jpg"))
    if len(existing) >= n:
        return existing[:n]
    urls = fetch_urls(dataset, n)
    tasks = [(u, d / f"{name}_{i:04d}.jpg") for i, u in enumerate(urls)]
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_download_one, tasks))
    got = sorted(d.glob("*.jpg"))
    print(f"  {name}: {len(got)} images")
    return got[:n]


def _items(paths: list[Path], label: int, gen: str) -> list[ImageItem]:
    return [ImageItem(path=p, label=label, generator=gen) for p in paths]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-generator drop experiment on GenImage")
    parser.add_argument("--per-class", type=int, default=250, help="images per class per split")
    parser.add_argument("--train-gen", default="midjourney", choices=list(GEN_SOURCES))
    parser.add_argument("--data-root", default="data/genimage_exp")
    parser.add_argument("--report", default=None, help="default: reports/cross_generator_<train-gen>.md")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args(argv)

    train_gen = args.train_gen
    report_path = args.report or f"reports/cross_generator_{train_gen}.md"
    seed_everything(0)
    device = get_device()
    pc = args.per_class
    root = Path(args.data_root)
    print(f"device {device} | {pc} images per class per split | train generator: {train_gen}")

    # Download: reals and the train generator need 2*pc (train + test halves); the
    # held-out generators only appear in test, so pc each.
    print("downloading (keyless, via datasets-server /rows) ...")
    reals = download_source("real", REAL_SOURCE, 2 * pc, root)
    gen_paths: dict[str, list[Path]] = {}
    for name, ds in GEN_SOURCES.items():
        need = 2 * pc if name == train_gen else pc
        gen_paths[name] = download_source(name, ds, need, root)

    # Split. Reals are shared: half train, half test. The train generator's fakes
    # split the same way; every other generator is test-only.
    real_train = _items(reals[:pc], 0, "real")
    real_test = _items(reals[pc : 2 * pc], 0, "real")
    train_fakes = _items(gen_paths[train_gen][:pc], 1, train_gen)
    test_sets: dict[str, list[ImageItem]] = {
        f"{train_gen} (same-gen)": real_test + _items(gen_paths[train_gen][pc : 2 * pc], 1, train_gen)
    }
    for name in GEN_SOURCES:
        if name == train_gen:
            continue
        test_sets[name] = real_test + _items(gen_paths[name][:pc], 1, name)

    transform = build_transforms(224, train=False)
    model = build_model("vit_small_patch16_224", "probe").to(device)

    # Carve a stratified val from train for best-head selection.
    train_items = real_train + train_fakes
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_items))
    n_val = max(2, int(0.2 * len(train_items)))
    val_items = [train_items[i] for i in perm[:n_val]]
    fit_items = [train_items[i] for i in perm[n_val:]]

    print("extracting frozen features for train/val ...")
    xtr, ytr = extract_all_features(model, fit_items, transform, device, 64)
    xva, yva = extract_all_features(model, val_items, transform, device, 64)
    best_val, _ = fit_head(
        classifier_module(model), xtr, ytr, xva, yva,
        epochs=args.epochs, lr=1e-3, weight_decay=1e-4, warmup_frac=0.1,
        batch_size=128, device=device, seed=0,
    )
    print(f"trained probe on {train_gen}: best val AUC {best_val:.4f}")

    import torch

    head = classifier_module(model).to(device).eval()

    def score(items: list[ImageItem]) -> tuple[np.ndarray, np.ndarray]:
        feats, labels = extract_all_features(model, items, transform, device, 64)
        with torch.no_grad():
            x = torch.from_numpy(feats).float().to(device)
            probs = torch.softmax(head(x), dim=1)[:, 1].float().cpu().numpy()
        return probs, labels

    results: dict[str, float] = {}
    fprs: dict[str, float] = {}
    for name, items in test_sets.items():
        s, y = score(items)
        results[name] = roc_auc(s, y)
        fprs[name] = fpr_at_tpr(s, y, 0.95)
        print(f"  {name}: AUC {results[name]:.4f} | FPR@TPR=0.95 {fprs[name]:.4f} ({len(items)} imgs)")

    same_key = f"{train_gen} (same-gen)"
    same_auc = results[same_key]
    cross = {k: v for k, v in results.items() if k != same_key}
    summary = cross_generator_drop(same_auc, cross)
    drop = summary["drop"]

    if drop > 0.02:
        verdict = (
            f"The detector loses {drop:.3f} AUC on average going to generators it never saw. That "
            "gap is the generalization cost a same-generator number hides."
        )
    elif drop < -0.02:
        verdict = (
            f"The unseen generators are actually easier ({-drop:.3f} AUC higher on average), not "
            f"harder. That happens when the training generator ({train_gen}) is a strong, modern "
            "model whose fakes are subtle, while the unseen ones are older models with blatant "
            "artifacts any detector flags. Cross-generator transfer is asymmetric: it depends on "
            "which generator is hard, not on seen-vs-unseen alone."
        )
    else:
        verdict = "The unseen and same generators score about the same here; no large gap either way."

    lines = [f"# Cross-generator drop: train on {train_gen} (GenImage)\n"]
    lines.append(
        f"A linear probe on a frozen `vit_small_patch16_224`, trained on the **{train_gen}** "
        f"generator with a shared real set (`{REAL_SOURCE}`), then tested on generators it never "
        f"saw. {pc} real and {pc} fake images per test set. The real distribution is identical "
        "across train and every test set, so the gap below is generator shift, not domain shift.\n"
    )
    lines.append("| test generator | ROC-AUC | FPR @ TPR=0.95 |")
    lines.append("| -------------- | ------- | -------------- |")
    lines.append(f"| {same_key} | {same_auc:.4f} | {fprs[same_key]:.4f} |")
    for name in cross:
        lines.append(f"| {name} (unseen) | {results[name]:.4f} | {fprs[name]:.4f} |")
    lines.append(
        f"\nMean unseen-generator AUC {summary['mean_cross_gen_auc']:.4f}, a change of "
        f"**{drop:+.4f}** (same-gen minus mean-unseen) from the same-generator AUC of "
        f"{same_auc:.4f}. {verdict}"
    )
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nreport written to {report}")


if __name__ == "__main__":
    main()
