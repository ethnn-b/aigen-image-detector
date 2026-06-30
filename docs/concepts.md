# Concepts

The ideas this project rests on, explained plainly. If you understand these, the code is
straightforward.

## What there is to detect

An AI image generator (a diffusion model like Stable Diffusion, or an older GAN) leaves traces. The
way it builds an image from noise puts regular patterns in the high frequencies, in the way edges
fall, and sometimes in small repeated textures the eye skips over. A detector does not need to
understand the picture. It needs to spot those traces. That is why a fairly ordinary classifier on
top of a pretrained backbone works at all: the features that separate cat from dog also happen to
carry the fingerprint of how the pixels were made.

The catch is that every generator leaves a different fingerprint. Train a detector on Stable
Diffusion and it learns Stable Diffusion's tells, not "fakeness" in general. Show it images from a
generator it never saw and the score falls. That gap is the whole subject of this project.

## Linear probe vs full fine-tune

Two ways to put a backbone to work:

- Linear probe. Freeze the pretrained backbone, take its feature vector for each image, and train
  only a small linear head on top. Fast, runs on a CPU, and a strong baseline. If the backbone's
  features already carry the generator fingerprint, the probe is enough.
- Full fine-tune. Unfreeze the backbone and train all of it on the real-vs-fake task. More accurate
  when the fingerprint is subtle, because the features themselves adapt. Needs a GPU, and it can
  overfit the one generator in the training data, which makes the cross-generator gap worse.

The project runs the probe first because it is cheap and honest, then the fine-tune, and compares
them. More compute is not automatically better here, and the comparison shows it.

## ROC-AUC, and why not just accuracy

Accuracy needs a threshold (call it fake above 0.5). Pick a different threshold and the accuracy
changes, so a single accuracy number hides how the detector behaves across thresholds. ROC-AUC fixes
that. It plots the true-positive rate against the false-positive rate as the threshold sweeps from 0
to 1, and takes the area under that curve. AUC of 1.0 is perfect, 0.5 is a coin flip. It is
threshold-free, so it compares two detectors fairly. That is the headline score here.

## FPR at a fixed TPR

AUC is one number for the whole curve. In practice you operate at one point on it, and the point that
matters for a fake detector is: set the threshold so you catch most of the fakes, then ask what that
costs in false alarms. So the operating metric is FPR at TPR = 0.95. Read it as: when the detector
catches 95 percent of AI images, what fraction of real photos does it wrongly flag as fake. A
detector with great AUC can still have an ugly FPR at the operating point, and that false alarm is
the error that hurts a real person. This is the number the report leads with.

## Cross-generator generalization

This is the honest test. Hold out whole generators. Train on the fakes from one generator (or a few),
and put only a different generator's fakes in the test set. Now the detector has to recognize a
fingerprint it never trained on. The AUC almost always drops, sometimes a lot. Reporting that drop,
instead of a same-generator number that looks great, is what separates a real evaluation from a demo.
The cross-generator table is the centerpiece of this project.

## Degradation under compression and resizing

The fingerprint lives partly in the high frequencies, and JPEG compression throws high frequencies
away. So a detector that aces clean PNGs can fall apart on the same images saved as JPEG, or shrunk
and re-enlarged. Since almost every image on the internet has been compressed at least once, this is
the realistic setting, not an edge case. The degradation curve re-saves the test images at a range of
JPEG qualities and at smaller sizes, and plots the AUC against that. A flat curve is a strong
detector; a cliff is a warning.

## Grad-CAM, to check the model is honest

A detector can score well for the wrong reason. If all the fake images in a dataset carry a small
watermark or a particular border, the model can learn the watermark instead of the generator
fingerprint, and it will fail the moment the watermark is gone. Grad-CAM produces a heatmap of which
pixels drove the decision. If the heat sits on a corner logo, the model found a shortcut. If it
spreads over textured regions, it is more likely reading the real signal. The heatmaps are evidence
for the claim that the detector learned something that generalizes, and a way to catch it when it
did not.

## Evaluation hygiene

A few rules so the numbers mean something:

- Split first, fit nothing on the test set. The threshold for the FPR report is chosen on validation,
  not on test.
- For the cross-generator table, the held-out generator must appear in no training image, or the gap
  is fake.
- Compare probe and fine-tune on the exact same split and the same metric.
- Report the operating-point FPR next to the AUC, because the AUC can look fine while the operating
  point does not.
