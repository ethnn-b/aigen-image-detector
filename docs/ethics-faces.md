# Ethics note for the faces track

Written before any face image is downloaded, on purpose. The framing has to be set
before the data is touched, not bolted on as a disclaimer at the end.

## What this is, and what it is not

The faces track runs the same detector on deepfake datasets (FaceForensics++ or DFDC).
It outputs a probability that a face image or video frame was manipulated. That number
is a signal, not proof. It does not identify a person, it does not establish intent,
and it cannot tell a satire clip from a malicious one. Treat a high score as a reason to
look closer with other evidence, never as a verdict on its own.

## The cost of a false positive

A real image detector that flags fakes has two errors. Calling a fake real lets a
manipulation through. Calling a real image fake accuses a real person of faking
something they did not. On faces, the second error is the one that hurts: it can brand a
genuine photo or video as fabricated and damage the person in it. That is why the whole
project reports FPR at a fixed TPR rather than a single accuracy number, and the faces
track is held to the same standard. A face detector that has not had its false-positive
rate measured on faces it never trained on should not be pointed at a real person.

## Generalization is worse on faces, not better

The central finding of this project, that a detector collapses on a generator it never
trained on, applies with more force to faces. Face manipulation methods change quickly,
and a detector trained on one family of fakes will under-perform on the next. So the
cross-generator and cross-method evaluation is not optional here. Any face result is
reported with the held-out-method AUC next to it, and the degradation curve too, because
a compressed social-media re-upload is the normal case for a face video.

## Data handling and consent

- FaceForensics++ and DFDC are used only under their own licenses and access terms.
  FaceForensics++ requires an access request; DFDC requires accepting the Kaggle rules.
  No face data is scraped, and none is redistributed from this repo.
- The datasets contain images of real people. They are used to measure detector behavior,
  not to build a database about any individual. No per-person identity labels are added,
  and no attempt is made to link faces to names.
- Example face images are not committed to the repository. Reports use aggregate numbers
  and, where an example is needed, a synthetic or already-public sample, not a dataset
  subject's face pulled out and published.

## Misuse this project does not support

This detector is a research and evaluation tool. It is not built for, and should not be
used for, surveillance, automated content takedowns without human review, scoring of
private individuals, or any setting where a wrong call carries a legal or reputational
penalty with no appeal. If the operating point has not been measured on data like the
data it will see, it is not ready to be used on people, and the honest answer is to say
so.
