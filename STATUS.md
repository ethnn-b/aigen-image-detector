# Status

Updated: 2026-06-30
Phase: in progress
Progress: 10/12 milestones

## Milestones
- [x] Config and the pure metric module in place, unit tested
- [x] CIFAKE downloaded and the train/val/test split built
- [x] Backbone + 2-class head builds, linear-probe mode
- [x] Training loop runs, best checkpoint saved by val AUC
- [x] Test accuracy and ROC-AUC reported
- [x] FPR-at-fixed-TPR and the worst false-positive examples
- [x] Full fine-tune mode, compared against the probe
- [ ] Cross-generator table on GenImage
- [x] Degradation curve under JPEG compression and resize
- [x] Grad-CAM heatmaps wired into infer
- [ ] Faces track on FaceForensics++ or DFDC, with the ethics note
- [x] Gradio demo

## Current state
Both the probe and the fine-tune run end to end on CIFAKE and the failure reports are real, not
placeholders. A vit_small_patch16_224 backbone, scored on the full 20,000-image CIFAKE test split,
trained on a 20k subset (18k train, 2k val) on MPS.

| setup            | accuracy | ROC-AUC | FPR @ TPR=0.95 |
| ---------------- | -------- | ------- | -------------- |
| probe            | 0.920    | 0.977   | 0.116          |
| fine-tune (2 ep) | 0.961    | 0.993   | 0.031          |

- Degradation, probe: AUC 0.975 clean, 0.853 at JPEG q=10, 0.816 at resize x0.25. The expected cliff.
- Degradation, fine-tune: 0.994 clean, 0.893 at JPEG q=10, 0.810 at resize x0.25. Ahead on clean,
  but no better than the probe under heavy resize, the extra accuracy does not buy robustness.
- 12 worst false positives per model copied to reports/false_positives/. Reports: reports/report.md
  (probe) and reports/report_finetune.md (fine-tune).

All modules are implemented: data (CIFAKE export + splits + transforms + degradation), model
(timm backbone, freeze/unfreeze, feature caching), engine (shared feature/score runners), train
(probe via cached features, fine-tune loop), evaluate (the failure report), explain (Grad-CAM for
both ViT tokens and CNN maps), infer (single image + heatmap), app (Gradio). 25 unit tests pass,
ruff is clean. The faces-track ethics note is written (docs/ethics-faces.md), ahead of any face data.

Checkpoints: best_probe.pt and best.pt are the probe (the documented baseline, matches report.md);
best_finetune.pt is the stronger same-generator model. Point evaluate/infer at it with --checkpoint.

## Blockers
None for the CIFAKE path. The two open milestones need external data:
- Cross-generator table needs GenImage downloaded under data/genimage/. The loader and the table
  code are built; evaluate.py prints the download link and skips cleanly when it is absent.
- Faces track needs FaceForensics++ access or the DFDC download. The pipeline is generator-agnostic,
  so it runs on those once the files are present.

## Next
1. Download GenImage and produce the cross-generator drop table, the headline result. The expectation
   to test: the fine-tune's same-generator edge shrinks or reverses on unseen generators.
2. Faces track on FaceForensics++ or DFDC, behind the ethics note already written.
3. A longer fine-tune on a GPU (this one was 2 epochs on MPS); see if more training widens or narrows
   the eventual cross-generator gap.
