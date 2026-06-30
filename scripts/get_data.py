"""Fetch the datasets.

CIFAKE is the one you can pull today with no key, through the Hugging Face
`datasets` library. The cross-generator set (GenImage) and the faces sets
(FaceForensics++, DFDC) need a manual download or an access request, so this script
prints the links rather than pretending it can fetch them. No scraping anywhere.

    uv run python scripts/get_data.py
    uv run python scripts/get_data.py --skip-cifake   # only print the rest
"""

from __future__ import annotations

import argparse
from pathlib import Path

OTHER_DATASETS = """\
Cross-generator and faces datasets (manual download or access request)

  GenImage (eight generators, for the cross-generator table)
    https://github.com/GenImage-Dataset/GenImage
    Download the per-generator folders you want, drop them under data/genimage/<generator>/.

  FaceForensics++ (deepfake faces, request access)
    https://github.com/ondyari/FaceForensics
    Fill the access form, then follow their download script into data/faceforensics/.

  Deepfake Detection Challenge (DFDC), on Kaggle
    https://www.kaggle.com/c/deepfake-detection-challenge
    Accept the rules, download, and unpack into data/dfdc/.
"""


def download_cifake(data_root: str) -> None:
    """Pull CIFAKE from the Hugging Face hub into data/cifake/."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("the `datasets` package is not installed.")
        print("install it with:  uv sync --extra data")
        print("or grab CIFAKE from Kaggle:")
        print("  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        return

    root = Path(data_root) / "cifake"
    root.mkdir(parents=True, exist_ok=True)
    print(f"downloading CIFAKE into {root} (real CIFAR-10 + Stable Diffusion fakes) ...")
    # The hub mirror exposes train/test splits with an image and a binary label.
    load_dataset("dragonintelligence/CIFAKE-image-dataset", cache_dir=str(root))
    print("CIFAKE ready. label 1 = fake, 0 = real.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch datasets")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--skip-cifake", action="store_true", help="only print the other links")
    args = parser.parse_args(argv)

    if not args.skip_cifake:
        download_cifake(args.data_root)

    print()
    print(OTHER_DATASETS)


if __name__ == "__main__":
    main()
