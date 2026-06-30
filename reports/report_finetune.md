# Failure report

Checkpoint `checkpoints/best_finetune.pt` (finetune, backbone `vit_small_patch16_224`, val AUC 0.9942). Test set: 20000 CIFAKE images.

## Headline

| metric | value |
| ------ | ----- |
| ROC-AUC | 0.9933 |
| accuracy @ 0.5 | 0.9612 |
| accuracy @ operating threshold | 0.9576 |
| FPR @ TPR=0.95 | 0.0313 |

Read the last row as: to catch 95% of the fakes, the detector wrongly flags 3.1% of real photos as fake.

## Worst false positives

Real images scored at or above the operating threshold (0.6883), worst first. 271 real images were flagged in all. Top 12 copied to `reports/false_positives/finetune` for inspection.

| rank | score | source image |
| ---- | ----- | ------------ |
| 1 | 0.9993 | test_018239.png |
| 2 | 0.9992 | test_018550.png |
| 3 | 0.9969 | test_015032.png |
| 4 | 0.9960 | test_010939.png |
| 5 | 0.9946 | test_018171.png |
| 6 | 0.9926 | test_014338.png |
| 7 | 0.9924 | test_017658.png |
| 8 | 0.9916 | test_013219.png |
| 9 | 0.9914 | test_011025.png |
| 10 | 0.9911 | test_017192.png |
| 11 | 0.9907 | test_018011.png |
| 12 | 0.9899 | test_013509.png |

## Degradation curve

AUC as the 4000 sampled test images are corrupted. A cliff means the
detector leans on high-frequency content that compression destroys.

| condition | ROC-AUC |
| --------- | ------- |
| clean | 0.9938 |
| jpeg q=90 | 0.9931 |
| jpeg q=70 | 0.9928 |
| jpeg q=50 | 0.9857 |
| jpeg q=30 | 0.9784 |
| jpeg q=10 | 0.8932 |
| resize x0.75 | 0.9670 |
| resize x0.5 | 0.9243 |
| resize x0.25 | 0.8104 |

## Cross-generator table

Skipped: no GenImage data found under `data/genimage/`. CIFAKE has a single generator (Stable Diffusion), so the cross-generator drop, the honest headline of this project, needs GenImage's eight generators. Get it from https://github.com/GenImage-Dataset/GenImage and drop per-generator folders under `data/genimage/<generator>/`, then re-run.
