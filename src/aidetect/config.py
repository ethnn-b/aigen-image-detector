"""Settings, read from the environment where it makes sense.

One place to change the backbone, the training mode, and the operating point the
false-positive report is read at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Pretrained backbone from timm. ViT-small by default; EfficientNet is the swap.
    backbone: str = os.environ.get("AIDETECT_BACKBONE", "vit_small_patch16_224")
    image_size: int = int(os.environ.get("AIDETECT_IMAGE_SIZE", "224"))

    # "probe" freezes the backbone and trains only the head. "finetune" trains all of it.
    mode: str = os.environ.get("AIDETECT_MODE", "probe")

    batch_size: int = int(os.environ.get("AIDETECT_BATCH", "64"))
    epochs: int = int(os.environ.get("AIDETECT_EPOCHS", "5"))
    # Probe trains the head fast; fine-tune nudges the backbone gently.
    lr: float = float(os.environ.get("AIDETECT_LR", "1e-3"))
    weight_decay: float = float(os.environ.get("AIDETECT_WD", "1e-4"))
    # Fraction of each warmup-then-cosine schedule spent warming up.
    warmup_frac: float = float(os.environ.get("AIDETECT_WARMUP", "0.1"))

    # Fraction of the train split held out for validation (best-checkpoint choice).
    val_fraction: float = float(os.environ.get("AIDETECT_VAL_FRAC", "0.1"))
    # Cap images per CIFAKE split; None uses the full set. The fast run sets this.
    limit_per_split: int | None = (
        int(os.environ["AIDETECT_LIMIT"]) if os.environ.get("AIDETECT_LIMIT") else None
    )

    data_root: str = os.environ.get("AIDETECT_DATA", "data")
    checkpoint_dir: str = os.environ.get("AIDETECT_CKPT", "checkpoints")

    # The false-positive report is read at this true-positive rate: catch this
    # fraction of fakes, then report how many real images that flags.
    operating_tpr: float = float(os.environ.get("AIDETECT_TPR", "0.95"))

    seed: int = int(os.environ.get("AIDETECT_SEED", "0"))

    def validate(self) -> None:
        if self.mode not in {"probe", "finetune"}:
            raise ValueError(f"mode must be 'probe' or 'finetune', got '{self.mode}'")
        if not (0.0 < self.operating_tpr <= 1.0):
            raise ValueError("operating_tpr must be in (0, 1]")
        if self.image_size < 32:
            raise ValueError("image_size looks too small")


def load() -> Config:
    cfg = Config()
    cfg.validate()
    return cfg
