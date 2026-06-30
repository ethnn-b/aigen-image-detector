"""Scoring, kept pure so it is unit tested.

Arrays in, numbers out. No torch in here. The whole failure report leans on these,
so they are the first thing written and the first thing pinned by tests. label 1
means fake (the positive class), label 0 means real.
"""

from __future__ import annotations

import numpy as np


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    return float((preds == labels).mean())


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean of their rank span. Same rule scipy uses."""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (counts[dense - 1] + counts[dense] + 1)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney) identity. Handles ties."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs at least one positive and one negative")
    ranks = _avg_ranks(scores)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_tpr(scores: np.ndarray, labels: np.ndarray, tpr_target: float = 0.95) -> float:
    """The score threshold that catches at least tpr_target of the fakes.

    To catch at least tpr_target of the positives, the threshold is the k-th highest
    positive score, where k = ceil(tpr_target * n_pos). This is chosen on validation
    and then reused on test, so the operating point is never fit on the test set.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    if len(pos) == 0:
        raise ValueError("threshold_at_tpr needs at least one positive")
    if not (0.0 < tpr_target <= 1.0):
        raise ValueError("tpr_target must be in (0, 1]")
    k = int(np.ceil(tpr_target * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(np.sort(pos)[::-1][k - 1])


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, tpr_target: float = 0.95) -> float:
    """Set the threshold to catch tpr_target of the fakes, return the fraction of real images flagged.

    This is the operating-point cost: when the detector catches most fakes, how
    often does it wrongly call a real photo fake.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    neg = scores[labels == 0]
    if len(neg) == 0:
        raise ValueError("fpr_at_tpr needs at least one negative")
    threshold = threshold_at_tpr(scores, labels, tpr_target)
    return float((neg >= threshold).mean())


def false_positive_indices(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> list[int]:
    """Indices of real images (label 0) scored at or above threshold, worst first.

    These are the images to look at in the report: the real photos the detector
    was most sure were fake.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    real = np.nonzero(labels == 0)[0]
    flagged = real[scores[real] >= threshold]
    return list(flagged[np.argsort(scores[flagged])[::-1]])


def cross_generator_drop(same_gen_auc: float, cross_gen_aucs: dict[str, float]) -> dict[str, float]:
    """Summarize how far AUC falls on generators not seen in training.

    Returns the mean cross-generator AUC and the drop from the same-generator AUC.
    A large drop is the honest headline of the project.
    """
    if not cross_gen_aucs:
        raise ValueError("need at least one held-out generator AUC")
    mean_cross = float(np.mean(list(cross_gen_aucs.values())))
    return {
        "same_gen_auc": float(same_gen_auc),
        "mean_cross_gen_auc": mean_cross,
        "drop": float(same_gen_auc - mean_cross),
    }
