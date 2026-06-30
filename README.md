# aigen-image-detector

Tell real photographs apart from AI-generated images. The model is a pretrained vision backbone with
a small head, trained as a linear probe or a full fine-tune. The part that matters is the report it
produces: not one accuracy number, but where the detector fails. Which real images it wrongly flags
as fake, how far its score drops on a generator it never trained on, and how quickly it breaks once
an image is compressed or resized.

> Status: the linear probe runs end to end on CIFAKE. Data pipeline, model, training loop, the
> failure report (FPR analysis, false-positive examples, degradation curve), Grad-CAM, single-image
> inference, and the Gradio demo are all built; tests pass. The cross-generator table is wired and
> waits only on GenImage being downloaded. See `STATUS.md` for the milestone-by-milestone state.

## The novel angle

Most AI-image detectors on student portfolios stop at "99 percent accuracy on the test set". That
number is close to meaningless, because the test set comes from the same generator as the training
set, and the real risk is the generator you have not seen yet. This project measures exactly that.
It trains on one source of fakes and tests on others, and it reports the drop instead of hiding it.

Three things the usual demo skips:

- A false-positive report. At a fixed detection rate (catch 95 percent of fakes), how many real
  photos get called fake, and which ones? A false accusation is the expensive error here.
- A cross-generator table. Train on Stable Diffusion images, test on a different generator, and show
  how much the score falls. This is the honest headline.
- A degradation curve. Re-save an image as JPEG or shrink it, and watch the detector degrade. Real
  images on the internet are compressed, so this is the realistic setting.

The faces track runs the same pipeline on deepfake datasets, with a short note on misuse and on what
a wrong call costs a person.

## Features

- A timm vision backbone (ViT or EfficientNet) with a 2-class head, run as a linear probe or a full
  fine-tune behind one config flag.
- A pure metrics module (accuracy, ROC-AUC, FPR at a fixed TPR, cross-generator drop) with unit
  tests, so the numbers are trustworthy.
- A cross-generator split that holds out whole generators for the test set.
- A degradation test under JPEG compression and resizing.
- Grad-CAM heatmaps so a flagged image shows where the model looked.
- A Gradio demo: upload an image, get real-or-fake with a confidence and a heatmap.

## Setup

Uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
uv sync --extra data    # adds the Hugging Face datasets loader for CIFAKE
```

The linear probe trains on CPU. The full fine-tune and the faces track want a GPU (Colab or Kaggle
both work).

## Usage

```bash
# pull CIFAKE (keyless via Hugging Face) and print links for the rest
uv run python scripts/get_data.py

# train the linear probe, then the full fine-tune
uv run python -m aidetect.train --mode probe
uv run python -m aidetect.train --mode finetune

# the failure report: accuracy, AUC, FPR at TPR=0.95, cross-generator table, degradation curve
uv run python -m aidetect.evaluate --checkpoint checkpoints/best.pt

# one image, with a Grad-CAM heatmap
uv run python -m aidetect.infer --image path/to/img.jpg --heatmap

# demo UI
uv run python -m aidetect.app
```

## Folder structure

```
aigen-image-detector/
  README.md             this file
  pyproject.toml
  docs/
    concepts.md         how AI-image detection works and how to evaluate it honestly
    design-decisions.md why CIFAKE, why FPR-at-TPR, why cross-generator is the real test
    ethics-faces.md     the false-accusation cost and misuse note for the faces track
  src/aidetect/         the package (config, data, model, metrics, train, evaluate, explain, infer, app)
  tests/                unit tests for the metric and split logic
  scripts/              data fetch helper
  data/                 datasets (gitignored)
  checkpoints/          saved models (gitignored)
```

## Datasets

All public, none scraped. The loaders read files the dataset hosts hand you.

- CIFAKE, real CIFAR-10 plus Stable Diffusion fakes (primary, keyless via Hugging Face):
  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- GenImage, eight generators, for the cross-generator table:
  https://github.com/GenImage-Dataset/GenImage
- FaceForensics++ (deepfake faces, request access): https://github.com/ondyari/FaceForensics
- DFDC on Kaggle: https://www.kaggle.com/c/deepfake-detection-challenge

## Results

Linear probe on CIFAKE: a frozen `vit_small_patch16_224` backbone, a 2-logit head trained on cached
features, scored on the full 20,000-image CIFAKE test split. Trained on a 20k subset of the train
split (18k train, 2k val) on an Apple-Silicon MPS device. The full report is in
[reports/report.md](reports/report.md).

| Setup                         | Accuracy | ROC-AUC | FPR @ TPR=0.95 |
| ----------------------------- | -------- | ------- | -------------- |
| Linear probe (CIFAKE)         | 0.920    | 0.977   | 0.116          |

Read the FPR column as the honest operating-point cost: to catch 95 percent of the AI images, the
probe wrongly flags 11.6 percent of real photos as fake (1,143 of 10,000 real test images). The 12
real images it was most sure were fake are copied to `reports/false_positives/`.

Degradation under JPEG compression and resizing, on a 4,000-image balanced sample of the test set:

| condition | clean | jpeg q=50 | jpeg q=10 | resize x0.5 | resize x0.25 |
| --------- | ----- | --------- | --------- | ----------- | ------------ |
| ROC-AUC   | 0.975 | 0.962     | 0.853     | 0.924       | 0.816        |

The AUC holds through light compression and falls off under heavy JPEG and aggressive downscaling,
which is the expected shape: a chunk of the generator fingerprint lives in the high frequencies that
those operations throw away.

Still to fill in: the full fine-tune row (the code path is built; run it on a GPU), and the
cross-generator table, which needs GenImage on disk (the loader and the table code are ready, and
`evaluate.py` prints the exact download instructions when GenImage is absent). The cross-generator
drop is the headline this project exists to show, and CIFAKE's single generator cannot produce it.
