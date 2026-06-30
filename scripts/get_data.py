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

OTHER_DATASETS = """\
Cross-generator and faces datasets (manual download or access request)

  GenImage (eight generators, for the cross-generator table)
    https://github.com/GenImage-Dataset/GenImage
    Drop each generator under data/genimage/<generator>/, with fake images in an
    `ai/` folder and real images in a `nature/` folder. GenImage's own train/val
    split inside (e.g. data/genimage/midjourney/val/ai, .../val/nature) is read as
    is. Then run:
      uv run python -m aidetect.evaluate --checkpoint checkpoints/best_probe.pt
    and the cross-generator table fills in. Tune coverage with --genimage-limit.

  FaceForensics++ (deepfake faces, request access)
    https://github.com/ondyari/FaceForensics
    Fill the access form, then follow their download script into data/faceforensics/.

  Deepfake Detection Challenge (DFDC), on Kaggle
    https://www.kaggle.com/c/deepfake-detection-challenge
    Accept the rules, download, and unpack into data/dfdc/.
"""


def download_cifake(data_root: str, limit_per_split: int | None) -> None:
    """Pull CIFAKE from the Hugging Face hub and export it to a real/fake image folder."""
    try:
        from aidetect.data import export_cifake
    except ImportError:
        print("could not import aidetect; run this through uv: uv run python scripts/get_data.py")
        return
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("the `datasets` package is not installed.")
        print("install it with:  uv sync --extra data")
        print("or grab CIFAKE from Kaggle:")
        print("  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        return

    note = "full set" if limit_per_split is None else f"{limit_per_split} per split"
    print(f"downloading + exporting CIFAKE into {data_root}/cifake/images ({note}) ...")
    images_root = export_cifake(data_root, limit_per_split=limit_per_split)
    print(f"CIFAKE ready under {images_root}. label 1 = fake, 0 = real.")
    print("train it:  uv run python -m aidetect.train --mode probe")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch datasets")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--skip-cifake", action="store_true", help="only print the other links")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap CIFAKE images per split (default: full 100k train / 20k test)",
    )
    args = parser.parse_args(argv)

    if not args.skip_cifake:
        download_cifake(args.data_root, args.limit)

    print()
    print(OTHER_DATASETS)


if __name__ == "__main__":
    main()
