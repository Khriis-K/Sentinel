"""
Sentinel — Shared Evaluation Protocol
Threshold tuning and ranked-event metrics used identically by the
next-event model and every baseline, so "are we beating the bar?" is
answerable: one tuning procedure, one metric set, test never touched
for tuning (ADR-0004).
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    """Precision among the top ``frac`` of ranked events (SOC review budget).

    Args:
        y_true: Binary labels (0 benign, 1 evil).
        scores: Anomaly scores, higher = more anomalous.
        frac: Fraction of events to review, e.g. 0.001 for 0.1%.

    Returns:
        Fraction of the top-``frac`` events that are evil.
    """
    n = len(scores)
    k = max(1, int(n * frac))
    k = min(k, n)
    top = np.argpartition(scores, n - k)[n - k:]
    return float(np.asarray(y_true)[top].mean())


def tune_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    n_candidates: int = 1000,
) -> Tuple[float, Dict[str, float]]:
    """Find the F1-maximizing threshold over the FULL score range.

    Candidates span [min(scores), max(scores)] — unlike the retired
    1st-99th-percentile sweep, the extremes remain reachable, so a model
    whose optimal cut sits at the tail is not silently excluded.

    Vectorized: predicted-positive count per candidate comes from a
    binary search over the sorted scores; true positives from a cumulative
    sum over the descending-score label order.

    Args:
        scores: Anomaly scores on the TUNING set (never test).
        y_true: Binary labels for the tuning set.
        n_candidates: Number of candidate thresholds.

    Returns:
        (threshold, metrics_at_threshold) where metrics has f1, precision,
        recall.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    n = len(scores)
    if n == 0:
        return 0.0, {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    lo, hi = float(scores.min()), float(scores.max())
    candidates = np.linspace(lo, hi, n_candidates) if hi > lo else np.array([lo])

    total_pos = int(y_true.sum())

    # Predicted-positive count per candidate: {scores >= t}
    s_asc = np.sort(scores)
    below = np.searchsorted(s_asc, candidates, side="left")  # count < t
    k = n - below  # count >= t
    k_safe = np.maximum(k, 1)

    # True positives at each k: cumulative evil count over descending scores
    order = np.argsort(-scores, kind="stable")
    cum_tp = np.cumsum(y_true[order])
    tp = cum_tp[k_safe - 1].astype(np.float64)
    tp[k == 0] = 0.0

    precision = tp / k_safe
    recall = tp / total_pos if total_pos > 0 else np.zeros_like(precision)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)

    best = int(np.argmax(f1))
    threshold = float(candidates[best])
    return threshold, {
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    """Compute the full honest-protocol metric set for given scores.

    Ranking metrics (AUROC, PR-AUC, precision@k) are threshold-free;
    f1/precision/recall sit at the provided (attack-val-tuned) threshold.

    Args:
        y_true: Binary labels (0 benign, 1 evil).
        scores: Anomaly scores, higher = more anomalous.
        threshold: Decision threshold tuned on the attack-val set.

    Returns:
        Dict with auroc, pr_auc, f1, precision, recall, confusion_matrix,
        threshold, precision_at_0_1 (0.1%), precision_at_1 (1%).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores >= threshold).astype(np.int64)

    if np.unique(y_true).size < 2:
        auroc = 0.0
        pr_auc = 0.0
    else:
        auroc = float(roc_auc_score(y_true, scores))
        pr_auc = float(average_precision_score(y_true, scores))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "threshold": float(threshold),
        "precision_at_0_1": precision_at_k(y_true, scores, 0.001),
        "precision_at_1": precision_at_k(y_true, scores, 0.01),
    }
