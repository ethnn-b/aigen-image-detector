"""Dataset loading, transforms, and the splits.

The cross-generator split helper is a pure function over file lists, so the
no-overlap guarantee is unit tested. CIFAKE arrives from the Hugging Face hub as
in-memory images; `export_cifake` writes them to disk as a real/fake image folder
so the rest of the code is uniformly path-based (the false-positive report needs
files it can open, and the degradation curve needs to re-save them as JPEG).

Heavy imports (torch, torchvision, PIL, datasets) are lazy so the pure split logic
stays importable without them, the same discipline metrics.py follows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Standard ImageNet statistics. Both the ViT and the EfficientNet timm backbones
# were pretrained with these, so the eval transform matches what they expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageItem:
    """One labelled image. label 1 = fake, 0 = real. generator names the source of a fake."""

    path: Path
    label: int
    generator: str = "real"


@dataclass(frozen=True)
class Splits:
    """A train/val/test partition of ImageItems. val is carved from CIFAKE's train
    split; test is CIFAKE's own held-out test split."""

    train: list[ImageItem]
    val: list[ImageItem]
    test: list[ImageItem]


def _stable_bucket(s: str, n: int) -> int:
    """A process-independent hash bucket. Python's builtin hash() is salted per run
    (PYTHONHASHSEED), so it cannot give the 'stable across runs' guarantee the split
    promises; md5 of the path can."""
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def cross_generator_split(
    items: list[ImageItem], test_generators: set[str]
) -> tuple[list[ImageItem], list[ImageItem]]:
    """Hold out whole generators for the test set.

    Every fake from a test generator goes to test, every other fake goes to train.
    Real images are split in half by a stable hash of their path so both sides have
    negatives without any path appearing twice. Raises if a test generator is also
    left in the train fakes, which would make the gap meaningless.
    """
    train: list[ImageItem] = []
    test: list[ImageItem] = []
    for it in items:
        if it.label == 1:
            (test if it.generator in test_generators else train).append(it)
        else:
            # deterministic real/real split, stable across runs, no path in both
            bucket = _stable_bucket(str(it.path), 2)
            (test if bucket == 0 else train).append(it)

    train_gens = {it.generator for it in train if it.label == 1}
    leaked = train_gens & test_generators
    if leaked:
        raise ValueError(f"test generators leaked into train: {sorted(leaked)}")
    return train, test


# --------------------------------------------------------------------------- #
# Transforms                                                                    #
# --------------------------------------------------------------------------- #


def build_transforms(image_size: int, train: bool = False):
    """Resize + ImageNet normalize for the chosen backbone.

    CIFAKE images are 32x32, so this is mostly an upscale to the backbone's input
    size. The probe caches features once, so it gets the plain eval transform; the
    fine-tune asks for `train=True` to add a horizontal flip.
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    steps = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def degrade_jpeg(img, quality: int):
    """Re-encode a PIL image as JPEG at the given quality, then decode it back.

    JPEG throws away high-frequency content, which is where a lot of the generator
    fingerprint lives, so this is the operation the degradation curve sweeps."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_resize(img, factor: float):
    """Downscale by `factor`, then scale back to the original size.

    factor 0.5 means halve the resolution and blow it back up, which smears the
    fine texture the detector keys on. factor 1.0 is a no-op."""
    from PIL import Image

    img = img.convert("RGB")
    if factor >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


# --------------------------------------------------------------------------- #
# A torch-style dataset over ImageItems                                         #
# --------------------------------------------------------------------------- #


class ImageItemDataset:
    """Indexable dataset of (image_tensor, label). A DataLoader only needs __len__
    and __getitem__, so this stays a plain class and data.py imports no torch.

    An optional `degrade` callable (PIL image -> PIL image) is applied before the
    transform, which is how the degradation curve feeds JPEG'd or shrunk images
    through the exact same model path.
    """

    def __init__(self, items: list[ImageItem], transform, degrade=None):
        self.items = items
        self.transform = transform
        self.degrade = degrade

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        it = self.items[i]
        img = Image.open(it.path).convert("RGB")
        if self.degrade is not None:
            img = self.degrade(img)
        return self.transform(img), it.label


# --------------------------------------------------------------------------- #
# CIFAKE: export the HF dataset to disk, then load it as ImageItems             #
# --------------------------------------------------------------------------- #

_CIFAKE_HF_ID = "dragonintelligence/CIFAKE-image-dataset"
# Words that mark the AI-generated class, matched case-insensitively against the
# dataset's ClassLabel names so we never guess which integer means "fake".
_FAKE_WORDS = ("fake", "ai", "synthetic", "generated", "gan")


def _detect_fake_index(class_names: list[str]) -> int:
    """Which ClassLabel index is the fake class, read from its name."""
    for i, name in enumerate(class_names):
        low = name.lower()
        if any(w in low for w in _FAKE_WORDS):
            return i
    raise ValueError(
        f"could not tell which CIFAKE label is fake from names {class_names}; "
        "inspect the dataset and set the mapping by hand"
    )


def export_cifake(
    data_root: str, limit_per_split: int | None = None, seed: int = 0
) -> Path:
    """Materialize CIFAKE from the HF hub into data/cifake/images/{train,test}/{real,fake}.

    Idempotent: a marker file records the export so a second run is a no-op. Returns
    the images root. `limit_per_split` caps how many images per HF split are written,
    which is what the fast end-to-end run uses; leave it None for the full set. When
    limiting, indices are sampled across the whole split (not a prefix) so both
    classes show up even if the split is stored class-ordered.
    """
    import numpy as np
    from datasets import load_dataset

    images_root = Path(data_root) / "cifake" / "images"
    marker = images_root / ".export_done.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta.get("limit_per_split") == limit_per_split:
            return images_root

    cache_dir = str(Path(data_root) / "cifake" / "hf_cache")
    ds = load_dataset(_CIFAKE_HF_ID, cache_dir=cache_dir)
    rng = np.random.default_rng(seed)

    counts: dict[str, int] = {}
    fake_index: int | None = None
    for split in ("train", "test"):
        if split not in ds:
            continue
        label_feat = ds[split].features["label"]
        names = getattr(label_feat, "names", None)
        if names is None:
            raise ValueError("CIFAKE label feature has no class names to read the mapping from")
        fake_index = _detect_fake_index(names)
        out_real = images_root / split / "real"
        out_fake = images_root / split / "fake"
        out_real.mkdir(parents=True, exist_ok=True)
        out_fake.mkdir(parents=True, exist_ok=True)
        total = len(ds[split])
        if limit_per_split is None or limit_per_split >= total:
            indices = range(total)
        else:
            indices = sorted(rng.permutation(total)[:limit_per_split].tolist())
        for i in indices:
            ex = ds[split][int(i)]
            is_fake = int(ex["label"]) == fake_index
            out = (out_fake if is_fake else out_real) / f"{split}_{i:06d}.png"
            if not out.exists():
                ex["image"].convert("RGB").save(out, format="PNG")
        counts[split] = len(indices) if not isinstance(indices, range) else total

    marker.write_text(
        json.dumps(
            {"limit_per_split": limit_per_split, "counts": counts, "fake_index": fake_index},
            indent=2,
        )
    )
    return images_root


def _list_split(images_root: Path, split: str) -> list[ImageItem]:
    items: list[ImageItem] = []
    for label, sub, gen in ((0, "real", "real"), (1, "fake", "stable_diffusion")):
        folder = images_root / split / sub
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.png")):
            items.append(ImageItem(path=p, label=label, generator=gen))
    return items


def load_cifake(
    data_root: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    limit_per_split: int | None = None,
) -> Splits:
    """Load CIFAKE as ImageItems with a train/val/test split.

    CIFAKE ships its own train and test splits; val is carved from train with a
    seeded, label-stratified shuffle so both classes are represented. The on-disk
    image folder is the source of truth: export runs only when it is empty, so a
    20k-subset export and a later full read do not fight over the marker. To rebuild
    at a different size, delete data/cifake/images and re-export.
    """
    import numpy as np

    images_root = Path(data_root) / "cifake" / "images"
    train_all = _list_split(images_root, "train")
    test = _list_split(images_root, "test")
    if not train_all or not test:
        export_cifake(data_root, limit_per_split=limit_per_split, seed=seed)
        train_all = _list_split(images_root, "train")
        test = _list_split(images_root, "test")
    if not train_all or not test:
        raise RuntimeError(
            f"no CIFAKE images under {images_root}; run scripts/get_data.py or export_cifake first"
        )

    # Stratified val carve-out: shuffle each class with the seed, take a fraction.
    rng = np.random.default_rng(seed)
    train: list[ImageItem] = []
    val: list[ImageItem] = []
    for label in (0, 1):
        group = [it for it in train_all if it.label == label]
        idx = rng.permutation(len(group))
        n_val = int(round(len(group) * val_fraction))
        val_idx = set(idx[:n_val].tolist())
        for j, it in enumerate(group):
            (val if j in val_idx else train).append(it)
    return Splits(train=train, val=val, test=test)
