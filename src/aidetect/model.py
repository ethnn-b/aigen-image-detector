"""The detector: a pretrained timm backbone with a 2-class head.

Freeze the backbone for probe mode, train all of it for fine-tune mode. The split
between probe and fine-tune is a single flag here so the rest of the code does not
care which ran: every checkpoint is the same timm model with a 2-logit head, and
evaluate, infer, and Grad-CAM load it the same way.

The probe trains fast because the frozen backbone's features are cached once (see
train.py). The two helpers for that, `extract_features` and `classifier_module`,
lean on timm's documented forward_features / forward_head(pre_logits=True) split,
which holds for both the ViT and the EfficientNet backbones.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_model(backbone: str, mode: str, pretrained: bool = True) -> nn.Module:
    """Return a timm backbone with a fresh 2-logit head.

    For mode == 'probe', every backbone parameter is frozen and only the head is
    left trainable. For mode == 'finetune', the whole model is trainable. label 1
    (fake) is logit index 1, label 0 (real) is index 0.
    """
    if mode not in {"probe", "finetune"}:
        raise ValueError(f"mode must be 'probe' or 'finetune', got '{mode}'")
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=2)
    set_trainable(model, mode)
    return model


def set_trainable(model: nn.Module, mode: str) -> None:
    """Freeze the backbone for a probe, leave everything on for a fine-tune."""
    if mode == "finetune":
        for p in model.parameters():
            p.requires_grad_(True)
        return
    # probe: freeze all, then re-enable just the classifier head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in classifier_module(model).parameters():
        p.requires_grad_(True)


def classifier_module(model: nn.Module) -> nn.Module:
    """The final Linear head. This is what the probe trains and what gets copied
    nowhere, since training it updates the model in place."""
    return model.get_classifier()


@torch.no_grad()
def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pooled pre-logits features for a batch, i.e. the input the head sees.

    `classifier_module(model)(extract_features(model, x))` reproduces the model's
    logits exactly, which is the contract the cached-feature probe relies on.
    """
    feats = model.forward_features(x)
    return model.forward_head(feats, pre_logits=True)


def feature_dim(model: nn.Module) -> int:
    """Width of the feature vector the head consumes."""
    head = classifier_module(model)
    if isinstance(head, nn.Linear):
        return head.in_features
    # Fall back to the attribute timm exposes on most backbones.
    return int(getattr(model, "num_features"))
