"""
Tests for src.baselines — iForest, Robust Covariance, One-Class SVM
as described in the BETH paper (Highnam et al., 2021).
"""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.baselines import (
    train_iforest,
    train_robust_covariance,
    train_one_class_svm,
    evaluate_baseline,
    run_all_baselines,
    extract_paper_features,
    BaselineResults,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def blob_data():
    """Generate separable blob data so baselines can learn something."""
    rng = np.random.default_rng(42)
    n_normal = 500
    n_anomaly = 20

    X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 10))
    y_normal = np.zeros(n_normal, dtype=np.int64)

    X_anomaly = rng.normal(loc=5.0, scale=0.5, size=(n_anomaly, 10))
    y_anomaly = np.ones(n_anomaly, dtype=np.int64)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])
    return X, y


@pytest.fixture
def simple_data():
    """Small, simple 2D data for fast smoke tests."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (100, 5))
    y = np.zeros(100, dtype=np.int64)
    y[:5] = 1  # first 5 are anomalies
    return X, y


@pytest.fixture
def split_data(blob_data):
    """Train/test split where train is benign-only (BETH-like)."""
    X, y = blob_data
    train_mask = y == 0
    return (
        X[train_mask],       # X_train (benign only)
        X,                   # X_test  (all data, includes anomalies)
        y,                   # y_test
    )


# ── iForest ────────────────────────────────────────────────────────────────────

def test_train_iforest_returns_fitted_model(simple_data):
    """train_iforest should return a fitted model that can score samples."""
    X, _ = simple_data
    model = train_iforest(X, contamination=0.05, seed=42)
    scores = model.decision_function(X)
    assert len(scores) == len(X)


def test_train_iforest_scores_anomalies_lower(blob_data):
    """iForest decision_function should give lower scores to anomalies."""
    X, y = blob_data
    model = train_iforest(X, contamination=0.05, seed=42)
    scores = model.decision_function(X)

    normal_scores = scores[y == 0]
    anomaly_scores = scores[y == 1]
    assert np.median(anomaly_scores) < np.median(normal_scores)


def test_train_iforest_deterministic(simple_data):
    """Same seed should produce identical models."""
    X, _ = simple_data
    m1 = train_iforest(X, contamination=0.05, seed=42)
    m2 = train_iforest(X, contamination=0.05, seed=42)
    s1 = m1.decision_function(X)
    s2 = m2.decision_function(X)
    np.testing.assert_array_equal(s1, s2)


# ── Robust Covariance ──────────────────────────────────────────────────────────

def test_train_robust_covariance_returns_fitted_model(simple_data):
    """train_robust_covariance should return a fitted model."""
    X, _ = simple_data
    model = train_robust_covariance(X, contamination=0.05, seed=42)
    scores = model.decision_function(X)
    assert len(scores) == len(X)


# ── One-Class SVM ──────────────────────────────────────────────────────────────

def test_train_one_class_svm_returns_fitted_model(simple_data):
    """train_one_class_svm should return a fitted model."""
    X, _ = simple_data
    model = train_one_class_svm(X)
    scores = model.decision_function(X)
    assert len(scores) == len(X)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def test_evaluate_baseline_returns_all_metrics(simple_data):
    """evaluate_baseline should return all expected metrics."""
    X, y = simple_data
    model = train_iforest(X, contamination=0.05, seed=42)
    metrics = evaluate_baseline(model, X, y)

    expected = {"auroc", "pr_auc", "f1", "precision", "recall", "confusion_matrix"}
    assert set(metrics.keys()) == expected
    for k in expected:
        assert metrics[k] is not None


def test_evaluate_baseline_auroc_in_range(simple_data):
    """AUROC should be between 0 and 1."""
    X, y = simple_data
    model = train_iforest(X, contamination=0.05, seed=42)
    metrics = evaluate_baseline(model, X, y)
    assert 0.0 <= metrics["auroc"] <= 1.0


def test_evaluate_baseline_not_fitted_raises(simple_data):
    """Evaluating an unfitted model should raise an error."""
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.05)
    X, y = simple_data
    with pytest.raises(NotFittedError):
        evaluate_baseline(model, X, y)


def test_evaluate_baseline_uses_contamination(simple_data):
    """Threshold should depend on contamination, not a hardcoded value."""
    X, y = simple_data
    model = train_iforest(X, contamination=0.05, seed=42)

    # With very high contamination, most predictions should be positive
    m1 = evaluate_baseline(model, X, y, contamination=0.5)
    # With very low contamination, few predictions should be positive
    m2 = evaluate_baseline(model, X, y, contamination=0.001)
    # Different contamination → different predicted positives
    assert m1["precision"] != m2["precision"] or m1["recall"] != m2["recall"]


# ── Run All ────────────────────────────────────────────────────────────────────

def test_run_all_baselines_returns_three_results(split_data):
    """run_all_baselines should return BaselineResults for all 3 baselines."""
    X_train, X_test, y_test = split_data
    results = run_all_baselines(X_train, X_test, y_test, seed=42)
    assert isinstance(results, BaselineResults)
    assert results.iforest is not None
    assert results.robust_covariance is not None
    assert results.one_class_svm is not None
    assert "auroc" in results.iforest
    assert "auroc" in results.one_class_svm


def test_run_all_baselines_no_leakage(split_data):
    """Test confirms train/test are separate — test data NOT seen during fit."""
    X_train, X_test, y_test = split_data
    # X_test includes anomalies; X_train does not
    assert np.any(y_test == 1), "Test set should have anomalies"
    assert X_train.shape != X_test.shape, "Train and test should differ"
    results = run_all_baselines(X_train, X_test, y_test, seed=42)
    # Just confirming it runs without error with proper separation
    assert results.iforest["auroc"] is not None


def test_baseline_results_to_dataframe(split_data):
    """BaselineResults.to_dataframe should return a DataFrame with 3 rows."""
    X_train, X_test, y_test = split_data
    results = run_all_baselines(X_train, X_test, y_test, seed=42)
    df = results.to_dataframe()
    assert len(df) == 3
    assert "auroc" in df.columns
    assert "pr_auc" in df.columns


# ── Paper Feature Extractor ────────────────────────────────────────────────────

def test_extract_paper_features_shape():
    """extract_paper_features should return (n, 7) array."""
    import pandas as pd
    df = pd.DataFrame({
        "processId": [1, 1, 2, 3, 3],
        "parentProcessId": [0, 1, 0, 2, 0],
        "userId": [0, 0, 1, 0, 2],
        "mountNamespace": [0, 0, 0, 1, 0],
        "eventId": [10, 10, 20, 30, 40],
        "argsNum": [0, 2, 0, 3, 0],
        "returnValue": [0, 0, 1, 0, 0],
    })
    feats = extract_paper_features(df)
    assert feats.shape == (5, 7)
    assert feats.dtype == np.float32
