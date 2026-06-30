"""Training loop. AdamW, linear warmup then cosine decay, best checkpoint by val ROC-AUC.

Two paths behind one flag:

- probe: freeze the backbone, extract its features once, and train only the linear
  head on the cached features. The expensive forward pass runs a single time, so
  the head fits in seconds even on a CPU.
- finetune: train the whole model end to end over the image loader, with a small
  learning rate so the pretrained features are nudged, not wrecked.

Both score validation ROC-AUC each epoch, keep the best weights, and save a
checkpoint that also records the operating-point threshold chosen on validation, so
evaluate.py never has to fit anything on the test set.
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as config_mod
from .data import build_transforms, load_cifake
from .engine import extract_all_features, predict_scores
from .metrics import roc_auc, threshold_at_tpr
from .model import build_model, classifier_module
from .util import get_device, seed_everything


def _make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    """Linear warmup to the base LR over the first warmup_frac of steps, then a
    cosine decay to zero. The standard recipe; keeps the head stable early."""
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def fit_head(
    head: nn.Module,
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    batch_size: int,
    device,
    seed: int = 0,
) -> tuple[float, dict]:
    """Train a linear head on cached features. Returns (best val AUC, best head state).

    Pure with respect to the backbone: it only sees feature arrays, so a test can
    feed synthetic separable features and a fresh nn.Linear, no timm needed.
    """
    seed_everything(seed)
    head = head.to(device)
    xtr = torch.from_numpy(np.ascontiguousarray(train_feats)).float()
    ytr = torch.from_numpy(np.ascontiguousarray(train_labels)).long()
    xva = torch.from_numpy(np.ascontiguousarray(val_feats)).float().to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(xtr)
    steps_per_epoch = max(1, math.ceil(n / batch_size))
    sched = _make_scheduler(opt, epochs * steps_per_epoch, warmup_frac)
    loss_fn = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_state = copy.deepcopy(head.state_dict())
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = xtr[idx].to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            val_scores = torch.softmax(head(xva), dim=1)[:, 1].float().cpu().numpy()
        auc = roc_auc(val_scores, val_labels)
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return best_auc, best_state


def _train_probe(model, splits, transform, cfg: config_mod.Config, device) -> float:
    print(f"extracting frozen features for {len(splits.train)} train / {len(splits.val)} val ...")
    tr_feats, tr_labels = extract_all_features(model, splits.train, transform, device, cfg.batch_size)
    va_feats, va_labels = extract_all_features(model, splits.val, transform, device, cfg.batch_size)
    print(f"training head: {tr_feats.shape[1]}-dim features, {cfg.epochs} epochs")
    best_auc, _ = fit_head(
        classifier_module(model),
        tr_feats,
        tr_labels,
        va_feats,
        va_labels,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_frac=cfg.warmup_frac,
        batch_size=256,
        device=device,
        seed=cfg.seed,
    )
    return best_auc


def _train_finetune(model, splits, transform, train_transform, cfg: config_mod.Config, device) -> float:
    from torch.utils.data import DataLoader

    from .data import ImageItemDataset

    model.to(device)
    train_ds = ImageItemDataset(splits.train, train_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = _make_scheduler(opt, cfg.epochs * max(1, len(train_loader)), cfg.warmup_frac)
    loss_fn = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
        val_scores, val_labels = predict_scores(model, splits.val, transform, device, cfg.batch_size)
        auc = roc_auc(val_scores, val_labels)
        print(f"epoch {epoch + 1}/{cfg.epochs}: val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_auc


def _save_checkpoint(path: Path, model, cfg: config_mod.Config, val_auc: float, val_threshold: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": cfg.backbone,
            "mode": cfg.mode,
            "image_size": cfg.image_size,
            "operating_tpr": cfg.operating_tpr,
            "val_auc": float(val_auc),
            "val_threshold": float(val_threshold),
            "classes": {"0": "real", "1": "fake"},
        },
        path,
    )


def train(mode: str, overrides: dict | None = None) -> None:
    base = config_mod.load()
    cfg = replace(base, mode=mode, **(overrides or {}))
    cfg.validate()
    seed_everything(cfg.seed)
    device = get_device()
    print(f"device: {device} | backbone: {cfg.backbone} | mode: {cfg.mode}")

    splits = load_cifake(
        cfg.data_root,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        limit_per_split=cfg.limit_per_split,
    )
    print(f"data: train {len(splits.train)} | val {len(splits.val)} | test {len(splits.test)}")

    model = build_model(cfg.backbone, cfg.mode).to(device)
    transform = build_transforms(cfg.image_size, train=False)

    if cfg.mode == "probe":
        best_auc = _train_probe(model, splits, transform, cfg, device)
    else:
        train_transform = build_transforms(cfg.image_size, train=True)
        best_auc = _train_finetune(model, splits, transform, train_transform, cfg, device)

    # Operating-point threshold is chosen on validation, then frozen into the
    # checkpoint, so evaluate.py reuses it on test without peeking.
    val_scores, val_labels = predict_scores(model, splits.val, transform, device, cfg.batch_size)
    threshold = threshold_at_tpr(val_scores, val_labels, cfg.operating_tpr)
    print(f"best val AUC {best_auc:.4f} | val threshold @ TPR={cfg.operating_tpr}: {threshold:.4f}")

    ckpt_dir = Path(cfg.checkpoint_dir)
    _save_checkpoint(ckpt_dir / f"best_{cfg.mode}.pt", model, cfg, best_auc, threshold)
    _save_checkpoint(ckpt_dir / "best.pt", model, cfg, best_auc, threshold)
    print(f"saved checkpoints to {ckpt_dir}/best_{cfg.mode}.pt and {ckpt_dir}/best.pt")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the AI-image detector")
    parser.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    parser.add_argument("--backbone", default=None, help="override the timm backbone")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap images per CIFAKE split")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.backbone is not None:
        overrides["backbone"] = args.backbone
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.limit is not None:
        overrides["limit_per_split"] = args.limit
    train(args.mode, overrides)


if __name__ == "__main__":
    main()
