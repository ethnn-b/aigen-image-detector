# Status

Updated: 2026-06-30
Phase: in progress
Progress: 9/12 milestones

## Milestones
- [x] Config and the pure metric module in place, unit tested
- [x] CIFAKE downloaded and the train/val/test split built
- [x] Backbone + 2-class head builds, linear-probe mode
- [x] Training loop runs, best checkpoint saved by val AUC
- [x] Test accuracy and ROC-AUC reported
- [x] FPR-at-fixed-TPR and the worst false-positive examples
- [ ] Full fine-tune mode, compared against the probe
- [ ] Cross-generator table on GenImage
- [x] Degradation curve under JPEG compression and resize
- [x] Grad-CAM heatmaps wired into infer
- [ ] Faces track on FaceForensics++ or DFDC, with the ethics note
- [x] Gradio demo

## Current state
The linear probe runs end to end on CIFAKE and the failure report is real, not a placeholder.
A frozen vit_small_patch16_224 backbone, head trained on cached features (MPS), scored on the full
20,000-image CIFAKE test split:

- ROC-AUC 0.977, accuracy 0.920.
- FPR @ TPR=0.95 = 0.116: catching 95% of fakes flags 11.6% of real photos (1,143 of 10,000).
- Degradation curve: AUC 0.975 clean, 0.853 at JPEG q=10, 0.816 at resize x0.25. The expected cliff.
- 12 worst false positives copied to reports/false_positives/. Full report in reports/report.md.

All modules are implemented: data (CIFAKE export + splits + transforms + degradation), model
(timm backbone, freeze/unfreeze, feature caching), engine (shared feature/score runners), train
(probe via cached features, fine-tune loop), evaluate (the failure report), explain (Grad-CAM for
both ViT tokens and CNN maps), infer (single image + heatmap), app (Gradio). 25 unit tests pass,
ruff is clean. The faces-track ethics note is written (docs/ethics-faces.md), ahead of any face data.

Checkpoints: best_probe.pt and best.pt are the probe; best_finetune.pt lands when that run finishes.

## Blockers
None for the probe path. The two open milestones need external data:
- Cross-generator table needs GenImage downloaded under data/genimage/. The loader and the table
  code are built; evaluate.py prints the download link and skips cleanly when it is absent.
- Faces track needs FaceForensics++ access or the DFDC download. The pipeline is generator-agnostic,
  so it runs on those once the files are present.

## Next
1. Run the full fine-tune on a GPU and fill the comparison row (the code path is built and tested).
2. Download GenImage and produce the cross-generator drop table, the headline result.
3. Faces track on FaceForensics++ or DFDC, behind the ethics note already written.
