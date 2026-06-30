"""Tests for the model wiring and the head-training step.

These build a backbone with pretrained=False so they need no network. ViT-tiny keeps
a forward pass cheap on CPU. The head-training test feeds synthetic separable
features, which is the injectable-fake-extractor path: it exercises train.fit_head
without timm being involved at all.
"""

import numpy as np
import torch
import torch.nn as nn

from aidetect.explain import gradcam
from aidetect.model import build_model, classifier_module, extract_features, feature_dim
from aidetect.train import fit_head

BACKBONE = "vit_tiny_patch16_224"


def test_probe_freezes_backbone_keeps_head():
    model = build_model(BACKBONE, "probe", pretrained=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"head.weight", "head.bias"}


def test_finetune_trains_everything():
    model = build_model(BACKBONE, "finetune", pretrained=False)
    assert all(p.requires_grad for p in model.parameters())


def test_extract_features_reproduces_logits():
    model = build_model(BACKBONE, "probe", pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = extract_features(model, x)
        logits_full = model(x)
        logits_head = classifier_module(model)(feats)
    assert feats.shape == (2, feature_dim(model))
    assert torch.allclose(logits_full, logits_head, atol=1e-5)


def test_fit_head_learns_separable_features():
    rng = np.random.default_rng(0)
    d = 16
    # two clusters: class 0 around -1, class 1 around +1, easily separable
    def make(n):
        x0 = rng.normal(-1.0, 0.3, size=(n, d))
        x1 = rng.normal(1.0, 0.3, size=(n, d))
        return np.vstack([x0, x1]), np.array([0] * n + [1] * n)

    xtr, ytr = make(200)
    xva, yva = make(50)
    head = nn.Linear(d, 2)
    best_auc, _ = fit_head(
        head, xtr, ytr, xva, yva,
        epochs=10, lr=1e-2, weight_decay=0.0, warmup_frac=0.1, batch_size=64, device="cpu", seed=0,
    )
    assert best_auc > 0.99


def test_gradcam_shape_and_range():
    model = build_model(BACKBONE, "probe", pretrained=False).eval()
    cam = gradcam(model, torch.randn(3, 224, 224), target_class=1)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
