"""Tests for the cross-generator path, the project's headline.

These verify the code that runs once GenImage is on disk, using a synthetic
GenImage-style folder and a random-weight backbone. The AUC values are meaningless
(the model is untrained) and nothing here is reported as a result; the point is that
the loader labels ai/nature correctly and the table renders a drop row, so the gated
path is proven before the real data lands.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from aidetect.data import build_transforms
from aidetect.evaluate import _balanced_subset, _load_generator_folder, cross_generator_table
from aidetect.model import build_model

BACKBONE = "vit_tiny_patch16_224"


def _write_genimage(root: Path, generator: str, n: int = 4) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for cls in ("ai", "nature"):
            d = root / "genimage" / generator / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{cls}_{i}.png")


def test_load_generator_folder_labels(tmp_path):
    _write_genimage(tmp_path, "midjourney", n=3)
    items = _load_generator_folder(tmp_path / "genimage" / "midjourney", "midjourney")
    fakes = [it for it in items if it.label == 1]
    reals = [it for it in items if it.label == 0]
    # ai -> fake (label 1, generator name kept), nature -> real (label 0)
    assert len(fakes) == 6 and len(reals) == 6  # 3 per class per split, two splits
    assert all(it.generator == "midjourney" for it in fakes)
    assert all(it.generator == "real" for it in reals)


def test_balanced_subset_is_balanced():
    from aidetect.data import ImageItem

    items = [ImageItem(Path(f"r{i}.png"), 0) for i in range(50)]
    items += [ImageItem(Path(f"f{i}.png"), 1, "g") for i in range(50)]
    sub = _balanced_subset(items, 20, seed=0)
    assert sum(it.label == 0 for it in sub) == 10
    assert sum(it.label == 1 for it in sub) == 10


def test_cross_generator_table_renders_drop(tmp_path):
    _write_genimage(tmp_path, "biggan", n=4)
    model = build_model(BACKBONE, "probe", pretrained=False).eval()
    transform = build_transforms(224, train=False)
    lines: list[str] = []
    cross_generator_table(
        model, transform, "cpu", 4, str(tmp_path), same_gen_auc=0.99, lines=lines, limit=0
    )
    text = "\n".join(lines)
    assert "biggan" in text
    assert "drop of" in text  # the summary line computed by cross_generator_drop


def test_cross_generator_table_skips_when_absent(tmp_path):
    model = build_model(BACKBONE, "probe", pretrained=False).eval()
    transform = build_transforms(224, train=False)
    lines: list[str] = []
    cross_generator_table(model, transform, "cpu", 4, str(tmp_path), 0.99, lines, limit=0)
    assert any("Skipped" in ln for ln in lines)
