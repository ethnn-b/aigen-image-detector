"""Run the model over a set of images: feature extraction and scoring.

Shared by train.py (probe feature caching, fine-tune validation) and evaluate.py
(test scoring, the degradation sweep), so the forward pass is written once. label 1
is the fake class, and the score returned everywhere is the softmax probability of
that class.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageItem, ImageItemDataset
from .model import extract_features


def _loader(items: list[ImageItem], transform, batch_size: int, degrade=None) -> DataLoader:
    ds = ImageItemDataset(items, transform, degrade=degrade)
    # num_workers=0 keeps this reliable on macOS/MPS; the images are tiny so the
    # serial decode is not the bottleneck, the backbone forward pass is.
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def extract_all_features(
    model, items: list[ImageItem], transform, device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen-backbone features for every item, in order. The probe trains its head
    on these, so the expensive forward pass happens once instead of every epoch."""
    model.eval()
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in _loader(items, transform, batch_size):
        x = x.to(device)
        f = extract_features(model, x)
        feats.append(f.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


@torch.no_grad()
def predict_scores(
    model, items: list[ImageItem], transform, device, batch_size: int = 64, degrade=None
) -> tuple[np.ndarray, np.ndarray]:
    """Fake-class probability and label for every item, in order. `degrade` (a
    PIL->PIL callable) injects the JPEG/resize corruption for the degradation curve
    through the exact same model path."""
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in _loader(items, transform, batch_size, degrade=degrade):
        x = x.to(device)
        prob_fake = torch.softmax(model(x), dim=1)[:, 1]
        scores.append(prob_fake.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(scores), np.concatenate(labels)
