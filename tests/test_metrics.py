import numpy as np
import pytest

from aidetect.metrics import (
    accuracy,
    cross_generator_drop,
    false_positive_indices,
    fpr_at_tpr,
    roc_auc,
    threshold_at_tpr,
)


def test_accuracy():
    assert accuracy(np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1])) == 0.75


def test_accuracy_empty():
    assert accuracy(np.array([]), np.array([])) == 0.0


def test_roc_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_inverted_is_zero():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_handles_ties():
    # two tied scores straddling the classes -> 0.5 contribution
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_needs_both_classes():
    with pytest.raises(ValueError):
        roc_auc(np.array([0.1, 0.2]), np.array([1, 1]))


def test_fpr_at_tpr_catches_all_fakes():
    # fakes score high, reals low. catching all fakes costs no false positives.
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert fpr_at_tpr(scores, labels, 1.0) == pytest.approx(0.0)


def test_fpr_at_tpr_overlap_costs_false_positives():
    # one real image scores above a fake. catching both fakes flags that real one.
    scores = np.array([0.3, 0.7, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    # to catch the 0.6 fake, threshold drops to 0.6, which flags the 0.7 real -> FPR 0.5
    assert fpr_at_tpr(scores, labels, 1.0) == pytest.approx(0.5)


def test_false_positive_indices_worst_first():
    scores = np.array([0.4, 0.95, 0.2, 0.7])
    labels = np.array([0, 0, 1, 0])  # indices 0,1,3 are real
    idx = false_positive_indices(scores, labels, threshold=0.5)
    # reals at or above 0.5 are index 1 (0.95) and index 3 (0.7), worst first
    assert idx == [1, 3]


def test_cross_generator_drop():
    out = cross_generator_drop(0.99, {"genA": 0.8, "genB": 0.7})
    assert out["mean_cross_gen_auc"] == pytest.approx(0.75)
    assert out["drop"] == pytest.approx(0.24)


def test_threshold_at_tpr_matches_fpr():
    # threshold chosen to catch all fakes drops to 0.6, so the 0.7 real is flagged.
    scores = np.array([0.3, 0.7, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    thr = threshold_at_tpr(scores, labels, 1.0)
    assert thr == pytest.approx(0.6)
    # and the FPR computed at that threshold is the same as fpr_at_tpr reports
    assert (scores[labels == 0] >= thr).mean() == pytest.approx(fpr_at_tpr(scores, labels, 1.0))
