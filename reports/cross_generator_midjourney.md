# Cross-generator drop: train on midjourney (GenImage)

A linear probe on a frozen `vit_small_patch16_224`, trained on the **midjourney** generator with a shared real set (`bitmind/bm-real`), then tested on generators it never saw. 250 real and 250 fake images per test set. The real distribution is identical across train and every test set, so the gap below is generator shift, not domain shift.

| test generator | ROC-AUC | FPR @ TPR=0.95 |
| -------------- | ------- | -------------- |
| midjourney (same-gen) | 0.9190 | 0.2840 |
| biggan (unseen) | 0.9914 | 0.0280 |
| adm (unseen) | 0.9997 | 0.0000 |
| glide (unseen) | 0.8956 | 0.5240 |

Mean unseen-generator AUC 0.9622, a change of **-0.0432** (same-gen minus mean-unseen) from the same-generator AUC of 0.9190. The unseen generators are actually easier (0.043 AUC higher on average), not harder. That happens when the training generator (midjourney) is a strong, modern model whose fakes are subtle, while the unseen ones are older models with blatant artifacts any detector flags. Cross-generator transfer is asymmetric: it depends on which generator is hard, not on seen-vs-unseen alone.
