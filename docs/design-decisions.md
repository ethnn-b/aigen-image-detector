# Design decisions

Each choice with the alternatives considered and why this one. Where a decision is reversible, that
is noted.

## What the project optimizes for

Decision: optimize for an honest failure report, not for peak accuracy on the training generator.

Most versions of this project chase a high number on a same-generator test set. That number is easy
and misleading. The decision here is to spend the effort on the cross-generator gap, the
operating-point false-positive rate, and how it holds up under compression, because those are the numbers
that say whether the detector would survive contact with a real image. The headline is allowed to be
a drop, as long as it is measured.

## Primary dataset: CIFAKE vs faces-first

Decision: CIFAKE as the primary dataset, faces as a later track.

- CIFAKE. Real CIFAR-10 images plus Stable Diffusion fakes, balanced, public, and downloadable with
  no access request.
  - Pros: anyone can run it today, balanced classes, small enough to train a probe on a CPU.
  - Cons: low resolution (32x32 upscaled), and one generator family, so it cannot show
    cross-generator generalization on its own.
- FaceForensics++ or DFDC first.
  - Pros: the deepfake-faces framing is what most people picture, and it is higher resolution.
  - Cons: FaceForensics++ needs an access request, DFDC is a large download, and faces add their own
    confounds. Starting here slows the first end-to-end run for no methodological gain.

CIFAKE gets the skeleton working fast and for free. GenImage is added for the cross-generator table
(its whole point is multiple generators), and the faces track comes after, with the ethics note.

## Backbone: ViT vs EfficientNet vs a from-scratch CNN

Decision: a pretrained timm backbone, ViT-small by default, EfficientNet as the swap.

- From-scratch CNN.
  - Pros: nothing to download, full control.
  - Cons: needs far more data and compute to match a pretrained model, and adds nothing to the point
    of the project.
- Pretrained ViT-small (timm).
  - Pros: strong features out of the box, the probe trains in minutes, good Grad-CAM story on the
    attention blocks.
  - Cons: heavier than a small CNN at inference.
- Pretrained EfficientNet.
  - Pros: light and fast, fine on a CPU.
  - Cons: slightly weaker features for this task in practice.

A pretrained backbone is the obvious call, since the project is about evaluation, not architecture.
ViT-small is the default for the feature quality and the cleaner heatmaps; EfficientNet is one config
line away if speed matters more.

## Probe first, fine-tune second

Decision: ship the linear probe before the full fine-tune.

The probe is cheap, runs on a CPU, and is a real baseline. The fine-tune is more accurate on the
training generator but tends to overfit that generator, which widens the cross-generator gap. Doing
both and comparing them is the interesting result: more compute does not always generalize better
here, and the numbers should show whether it did.

## Headline metric: FPR-at-TPR vs raw accuracy

Decision: report ROC-AUC plus FPR at TPR = 0.95, not accuracy alone.

Accuracy hides the threshold and the class costs. For a fake detector the expensive mistake is
flagging a real photo as fake, so the operating metric is set to catch 95 percent of fakes and then
report how many real images that costs. AUC gives the threshold-free summary; the fixed-TPR point
gives the number a user would actually feel. Accuracy is logged as context, not as the headline.

## Cross-generator split: held-out generators vs a random split

Decision: hold out whole generators for the test set.

A random split lets the same generator's fingerprint appear in train and test, so the model can pass
by memorizing that one fingerprint. Holding out entire generators forces it to generalize, which is
the real-world condition (the next generator is always one you have not trained on). The split helper
is a pure function over file lists, so the no-overlap guarantee is unit tested, the same discipline
the isl-recognition repo uses for its signer-independent split.

## Explainability: Grad-CAM vs nothing

Decision: include Grad-CAM heatmaps.

A detector can hit a high score by learning a shortcut (a watermark, a border, a compression
artifact unique to one class's source). Grad-CAM shows which pixels drove the call, which is how you
catch that. It costs little and it backs up the central claim that the model learned a real,
transferable signal. Without it, "the model generalizes" is an assertion; with it, there is a picture
to argue from.

## Ethics note for the faces track

Decision: write the ethics note before downloading any face data.

Deepfake detection touches real people, and a false accusation is a real harm. The note states the
false-positive cost in plain terms, says the detector is a signal and not proof, and records what the
dataset licenses do and do not allow. Writing it first keeps the framing honest instead of bolting a
disclaimer on at the end.
