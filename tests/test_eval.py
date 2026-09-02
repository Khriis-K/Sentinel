"""
Tests for src.eval — shared threshold tuning and ranked metrics.
These power both the next-event model and the iForest re-baseline, so
they must agree on protocol: candidates span the FULL score range.
"""
import numpy as np
import pytest

from src.eval import evaluate_scores, precision_at_k, tune_threshold


# ── precision@k ───────────────────────────────────────────────────────────────

def test_precision_at_k_top_block():
    """Perfect ranking: top 1% of a 1000-event set holds all 10 evils."""
    scores = np.arange(1000, dtype=np.float64)      # ascending
    y_true = np.zeros(1000, dtype=np.int64)
    y_true[-10:] = 1                                 # evils at the top
    assert precision_at_k(y_true, scores, 0.01) == 1.0


def test_precision_at_k_bottom_block():
    """Inverted ranking: top 1% holds no evils."""
    scores = np.arange(1000, dtype=np.float64)
    y_true = np.zeros(1000, dtype=np.int64)
    y_true[:10] = 1
    assert precision_at_k(y_true, scores, 0.01) == 0.0


def test_precision_at_k_half():
    """Half-precision case rounds through the actual top-k set."""
    scores = np.arange(100, dtype=np.float64)
    y_true = np.zeros(100, dtype=np.int64)
    y_true[-5:] = 1
    # top 10% = 10 events, 5 of them evil
    assert precision_at_k(y_true, scores, 0.1) == 0.5


def test_precision_at_k_small_n_at_least_one():
    """0.1% of a 100-event set rounds up to at least 1 reviewed event."""
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    y_true = np.array([1, 0, 0, 0, 0])
    assert precision_at_k(y_true, scores, 0.001) == 1.0


# ── Threshold tuning ──────────────────────────────────────────────────────────

def test_tune_threshold_finds_perfect_separation():
    """With separable scores, F1 = 1.0 at a threshold inside the gap."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95, 1.0])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1])
    threshold, metrics = tune_threshold(scores, y_true)
    assert metrics["f1"] == 1.0
    assert 0.4 < threshold <= 0.9


def test_tune_threshold_candidates_span_full_range():
    """The best threshold may sit at the extreme end — 1st-99th percentile
    sweeping would miss it; full-range candidates must find it.

    The only evil event has the LOWEST score, so the F1-optimal cut is to
    flag everything (threshold at the minimum) — P=1/4, R=1 → F1=0.4,
    and any higher threshold misses the evil event entirely (F1=0).
    """
    scores = np.array([100.0, 1.0, 2.0, 3.0])
    y_true = np.array([0, 1, 0, 0])  # evil event has the LOWEST score
    threshold, metrics = tune_threshold(scores, y_true)
    assert threshold <= scores.min() + 1e-9
    assert metrics["f1"] == pytest.approx(0.4)
    assert metrics["recall"] == 1.0


def test_tune_threshold_constant_scores():
    """All-identical scores must not crash and must flag everything."""
    scores = np.full(10, 2.5)
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    threshold, metrics = tune_threshold(scores, y_true)
    assert threshold == 2.5
    assert metrics["recall"] == 1.0  # everything predicted positive


def test_tune_threshold_deterministic():
    """Same inputs must give the same threshold."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=500)
    y_true = (rng.random(500) < 0.1).astype(np.int64)
    t1, m1 = tune_threshold(scores, y_true)
    t2, m2 = tune_threshold(scores, y_true)
    assert t1 == t2 and m1 == m2


def test_tune_threshold_no_positives():
    """All-benign tuning set must return a threshold without crashing."""
    scores = np.arange(50, dtype=np.float64)
    y_true = np.zeros(50, dtype=np.int64)
    threshold, metrics = tune_threshold(scores, y_true)
    assert metrics["f1"] == 0.0


# ── evaluate_scores ───────────────────────────────────────────────────────────

def test_evaluate_scores_keys():
    """The metric dict must carry the full honest-protocol key set."""
    rng = np.random.default_rng(1)
    scores = rng.normal(size=200)
    y_true = (rng.random(200) < 0.2).astype(np.int64)
    metrics = evaluate_scores(y_true, scores, threshold=0.5)

    expected = {
        "auroc", "pr_auc", "f1", "precision", "recall",
        "confusion_matrix", "threshold", "precision_at_0_1", "precision_at_1",
    }
    assert expected == set(metrics.keys())


def test_evaluate_scores_perfect_classifier():
    """Perfectly separated scores must yield perfect metrics at the right cut."""
    scores = np.concatenate([np.zeros(990), np.ones(10)])
    y_true = np.concatenate([np.zeros(990, dtype=np.int64), np.ones(10, dtype=np.int64)])
    metrics = evaluate_scores(y_true, scores, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision_at_0_1"] == 1.0  # top 0.1% of 1000 = 1 event, evil
    assert metrics["precision_at_1"] == 1.0    # top 1% of 1000 = 10 events, all evil


def test_evaluate_scores_confusion_matrix_shape():
    """Confusion matrix must always be 2x2 with labels=[0, 1]."""
    scores = np.array([0.1, 0.9])
    y_true = np.array([1, 0])  # both classes present
    metrics = evaluate_scores(y_true, scores, threshold=0.5)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm[0, 1] == 1  # benign (0) predicted positive
    assert cm[1, 0] == 1  # evil (1) predicted negative


def test_evaluate_scores_single_class_labels():
    """Single-class labels must not crash sklearn — metrics degrade to 0."""
    scores = np.array([0.1, 0.2, 0.3])
    y_true = np.zeros(3, dtype=np.int64)
    metrics = evaluate_scores(y_true, scores, threshold=0.2)
    assert metrics["auroc"] == 0.0
    assert metrics["pr_auc"] == 0.0
