# Failure report

Checkpoint `checkpoints/best.pt` (probe, backbone `vit_small_patch16_224`, val AUC 0.9767). Test set: 20000 CIFAKE images.

## Headline

| metric | value |
| ------ | ----- |
| ROC-AUC | 0.9766 |
| accuracy @ 0.5 | 0.9205 |
| accuracy @ operating threshold | 0.9173 |
| FPR @ TPR=0.95 | 0.1161 |

Read the last row as: to catch 95% of the fakes, the detector wrongly flags 11.6% of real photos as fake.

## Worst false positives

Real images scored at or above the operating threshold (0.4057), worst first. 1143 real images were flagged in all. Top 12 copied to `reports/false_positives` for inspection.

| rank | score | source image |
| ---- | ----- | ------------ |
| 1 | 0.9967 | test_017192.png |
| 2 | 0.9946 | test_010939.png |
| 3 | 0.9937 | test_018550.png |
| 4 | 0.9927 | test_018339.png |
| 5 | 0.9924 | test_015189.png |
| 6 | 0.9872 | test_018239.png |
| 7 | 0.9863 | test_016899.png |
| 8 | 0.9853 | test_010689.png |
| 9 | 0.9817 | test_017658.png |
| 10 | 0.9810 | test_019726.png |
| 11 | 0.9785 | test_016441.png |
| 12 | 0.9743 | test_016369.png |

## Degradation curve

AUC as the 4000 sampled test images are corrupted. A cliff means the
detector leans on high-frequency content that compression destroys.

| condition | ROC-AUC |
| --------- | ------- |
| clean | 0.9753 |
| jpeg q=90 | 0.9743 |
| jpeg q=70 | 0.9749 |
| jpeg q=50 | 0.9620 |
| jpeg q=30 | 0.9523 |
| jpeg q=10 | 0.8529 |
| resize x0.75 | 0.9548 |
| resize x0.5 | 0.9239 |
| resize x0.25 | 0.8163 |

## Cross-generator table

Skipped: no GenImage data found under `data/genimage/`. CIFAKE has a single generator (Stable Diffusion), so the cross-generator drop, the honest headline of this project, needs GenImage's eight generators. Get it from https://github.com/GenImage-Dataset/GenImage and drop per-generator folders under `data/genimage/<generator>/`, then re-run.
