# Cross-generator drop: train on biggan (GenImage)

A linear probe on a frozen `vit_small_patch16_224`, trained on the **biggan** generator with a shared real set (`bitmind/bm-real`), then tested on generators it never saw. 250 real and 250 fake images per test set. The real distribution is identical across train and every test set, so the gap below is generator shift, not domain shift.

| test generator | ROC-AUC | FPR @ TPR=0.95 |
| -------------- | ------- | -------------- |
| biggan (same-gen) | 0.9319 | 0.1760 |
| midjourney (unseen) | 0.8351 | 0.7480 |
| adm (unseen) | 0.8763 | 0.5160 |
| glide (unseen) | 0.9101 | 0.4320 |

Mean unseen-generator AUC 0.8738, a change of **+0.0581** (same-gen minus mean-unseen) from the same-generator AUC of 0.9319. The detector loses 0.058 AUC on average going to generators it never saw. That gap is the generalization cost a same-generator number hides.
