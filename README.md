# aigen-image-detector

A real-vs-AI image detector you can actually try. Upload a photo and it tells you whether it's a real
photograph or AI-generated, how confident it is, and — as a Grad-CAM heatmap — where it looked to
decide.

The project is the demo: a Gradio app on top of a fine-tuned vision model. Behind it, the detector
was put through an honest failure analysis (see [What I tested](#what-i-tested)), because one accuracy
number hides the errors that actually matter.

## The demo

```bash
uv run python -m aidetect.app
```

Upload an image and the app returns three things: the probability it's AI-generated, a real-or-fake
call at the model's operating threshold, and a Grad-CAM overlay of the regions that drove the call.

![A real photo scored real with 98 percent confidence](docs/screenshots/demo-real-photo.png)

A real photograph, scored **real at 98% confidence**. The heatmap spreads across the whole frame
rather than latching onto a single artifact.

![A fake-leaning score held to a real call by the operating threshold](docs/screenshots/demo-operating-threshold.png)

The operating threshold at work: this image's raw score leans fake (61%), but that sits below the
checkpoint's 0.688 threshold, so the call stays **real**. The confidence bar and the verdict can
legitimately disagree near the boundary — and the demo uses the very same threshold the evaluation
does, so "fake" means the same thing in both places.

## How it works

- **Backbone + head.** A timm `vit_small_patch16_224` with a small 2-class head, trained either as a
  linear probe (backbone frozen, head trained on cached features — runs on CPU) or a full fine-tune
  (whole model, wants a GPU), switched by one config flag.
- **Grad-CAM.** Any prediction can render a heatmap, so a flagged image shows *where* the model looked.
- **One checkpoint, two uses.** The same trained checkpoint powers both the demo and the evaluation,
  so what the demo calls fake is what the report measured.

![Pipeline](docs/diagrams/pipeline.svg)

## Run it yourself

Uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
uv sync --extra data                          # Hugging Face datasets loader for CIFAKE

uv run python scripts/get_data.py             # pull CIFAKE (keyless)
uv run python -m aidetect.train --mode probe  # or --mode finetune (GPU)
uv run python -m aidetect.app                 # launch the demo

# or score one image from the CLI, with a heatmap
uv run python -m aidetect.infer --image path/to/img.jpg --heatmap
```

## What I tested

"99% on the test set" is close to meaningless when the test images come from the same generator as
training. So beyond headline accuracy I measured the three things that decide whether a detector is
usable in the wild.

**Accuracy and the false-positive cost** — full CIFAKE test set (`vit_small_patch16_224`):

| Setup          | Accuracy | ROC-AUC | FPR @ TPR=0.95 |
| -------------- | -------- | ------- | -------------- |
| Linear probe   | 0.920    | 0.977   | 0.116          |
| Full fine-tune | 0.961    | 0.993   | 0.031          |

Read the last column as the operating-point cost: to catch 95% of fakes, the fine-tune wrongly flags
3.1% of real photos as fake (the probe, 11.6%). A false accusation is the expensive error here. Full
reports: [probe](reports/report.md), [fine-tune](reports/report_finetune.md).

**Cross-generator transfer** — the real test. Train on one generator, test on generators never seen
(GenImage, with a shared real set so only the generator shifts). Transfer is *asymmetric*: train on
modern MidJourney and older generators are actually easier (mean unseen AUC 0.962 vs 0.919 on
MidJourney's own test set); train on BigGAN and it collapses on unseen MidJourney (AUC 0.835 — where
catching 95% of fakes would flag roughly three of every four real photos). A detector is only as good
as the hardest generator it trained against. Reports:
[midjourney](reports/cross_generator_midjourney.md), [biggan](reports/cross_generator_biggan.md).

**Degradation** — real internet images are compressed and resized. AUC holds through light JPEG and
falls off under heavy compression and aggressive downscaling (fine-tune: 0.994 clean → 0.810 at resize
×0.25), because much of the generator fingerprint lives in the high frequencies those operations throw
away.

## Datasets

All public, none scraped.

- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) —
  real CIFAR-10 + Stable Diffusion fakes (primary, keyless via Hugging Face)
- [GenImage](https://github.com/GenImage-Dataset/GenImage) — eight generators, for the cross-generator test
- [FaceForensics++](https://github.com/ondyari/FaceForensics) and
  [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) — deepfake faces (request access)

## Layout

```
src/aidetect/   config, data, model, metrics, train, evaluate, explain, infer, app
docs/           concepts, design decisions, ethics note, diagrams, screenshots
reports/        generated failure reports + false-positive examples
scripts/        data fetch + cross-generator experiment
tests/          unit tests for the metric and split logic
```
