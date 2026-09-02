"""
Tests for src.baselines — iForest, Robust Covariance, One-Class SVM
as described in the BETH paper (Highnam et al., 2021), plus the
re-baselined protocol in src.baseline_protocol (attack-val thresholds,
precision@k, transductive caveat).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.baseline_protocol import BASELINE_CAVEAT, run_baseline_comparison
from src.baselines import (
    train_iforest,
    train_robust_covariance,
    train_one_class_svm,
    score_baseline,
    extract_paper_features,
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
    np.testing.assert_array_equal(m1.decision_function(X), m2.decision_function(X))


# ── Robust Covariance / One-Class SVM ─────────────────────────────────────────

def test_train_robust_covariance_returns_fitted_model(simple_data):
    """train_robust_covariance should return a fitted model."""
    X, _ = simple_data
    model = train_robust_covariance(X, contamination=0.05, seed=42)
    scores = model.decision_function(X)
    assert len(scores) == len(X)


def test_train_one_class_svm_returns_fitted_model(simple_data):
    """train_one_class_svm should return a fitted model."""
    X, _ = simple_data
    model = train_one_class_svm(X)
    scores = model.decision_function(X)
    assert len(scores) == len(X)


# ── Raw Scores ─────────────────────────────────────────────────────────────────

def test_score_baseline_inverts_decision_function(simple_data):
    """score_baseline must return HIGHER = more anomalous."""
    X, y = simple_data
    model = train_iforest(X, contamination=0.05, seed=42)
    scores = score_baseline(model, X)

    raw = model.decision_function(X)
    np.testing.assert_allclose(scores, -raw)
    assert scores.dtype == np.float64


def test_score_baseline_anomalies_score_higher(blob_data):
    """With the inversion, anomalies must score HIGHER than normals."""
    X, y = blob_data
    model = train_iforest(X, contamination=0.05, seed=42)
    scores = score_baseline(model, X)
    assert np.median(scores[y == 1]) > np.median(scores[y == 0])


def test_score_baseline_unfitted_raises(simple_data):
    """Scoring an unfitted model should raise."""
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.05)
    X, _ = simple_data
    with pytest.raises(NotFittedError):
        score_baseline(model, X)


# ── Paper Feature Extractor ────────────────────────────────────────────────────

def test_extract_paper_features_shape():
    """extract_paper_features should return (n, 7) array."""
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


def test_extract_paper_features_without_mount_namespace():
    """Per-host CSVs lack mountNamespace — the extractor must not KeyError."""
    df = pd.DataFrame({
        "processId": [1, 2],
        "parentProcessId": [0, 1],
        "userId": [0, 1],
        "eventId": [10, 20],
        "argsNum": [0, 2],
        "returnValue": [0, 1],
    })
    feats = extract_paper_features(df)
    assert feats.shape == (2, 7)
    assert (feats[:, 3] == 0).all()  # mountNamespace column zero-filled


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Protocol Tests (re-baseline under the honest protocol)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_baseline_csv_dir(tmp_path):
    """Tiny per-host corpus: 3 benign + 2 evil hosts with evil bursts."""
    per_host = tmp_path / "per_host"
    per_host.mkdir()
    ts = 0.0
    rng = np.random.default_rng(3)
    for host, evil_slice in [
        ("benign-a", None),
        ("benign-b", None),
        ("benign-c", None),
        ("evil-1", (400, 440)),
        ("evil-2", (600, 640)),
    ]:
        n = 1200
        evil = np.zeros(n, dtype=np.int64)
        if evil_slice:
            evil[evil_slice[0]:evil_slice[1]] = 1
        df = pd.DataFrame({
            "timestamp": np.arange(ts, ts + n, dtype=np.float64),
            "processId": rng.integers(1, 200, n),
            "parentProcessId": rng.integers(0, 3, n),
            "userId": rng.choice([0, 0, 0, 1], n),
            "processName": rng.choice(["systemd", "bash"], n),
            "hostName": [host] * n,
            "eventId": rng.integers(1, 15, n),
            "eventName": ["execve"] * n,
            "argsNum": rng.integers(0, 4, n),
            "returnValue": rng.choice([0, 0, 0, -1], n),
            "args": ["-c ls"] * n,
            "sus": np.zeros(n, dtype=np.int64),
            "evil": evil,
        })
        ts += n + 1.0
        df.to_csv(per_host / f"{host}.csv", index=False)
    return str(per_host)


def test_run_baseline_comparison_structure(tmp_path):
    """Comparison must return metrics per baseline plus the caveat."""
    data_dir = _make_baseline_csv_dir(tmp_path)
    results = run_baseline_comparison(
        data_dir, train_sample_size=500, svm_sample_size=200, seed=42,
    )

    for name in ["iforest", "robust_covariance", "one_class_svm"]:
        metrics = results[name]
        for key in ["auroc", "pr_auc", "f1", "precision", "recall",
                    "precision_at_0_1", "precision_at_1", "confusion_matrix",
                    "threshold", "threshold_tuning"]:
            assert key in metrics, f"{name} missing {key}"
        assert metrics["threshold_tuning"]["tune_source"] == "attack-val"
    assert results["caveat"] == BASELINE_CAVEAT


def test_run_baseline_comparison_matches_neural_carve(tmp_path):
    """Given the same seed, baselines must be scored on exactly the carved
    test corpus the neural pipeline derives (deterministic split + carve)."""
    from src.data import carve_attack_val, load_beth_data, split_by_host

    data_dir = _make_baseline_csv_dir(tmp_path)
    results = run_baseline_comparison(
        data_dir, train_sample_size=500, svm_sample_size=200, seed=11,
    )

    # Re-derive the same deterministic carve the protocol consumed
    host_dfs = load_beth_data(data_dir)
    parts = [df.sort_values("timestamp") for _, df in sorted(host_dfs.items())]
    _, _, test_pre = split_by_host(pd.concat(parts, ignore_index=True), seed=11)
    _, test_df = carve_attack_val(test_pre, n_blocks=50, tune_frac=0.2, seed=11)

    cm = np.array(results["iforest"]["confusion_matrix"])
    assert int(cm.sum()) == len(test_df)
    assert int(cm[1, :].sum()) == int(test_df["evil"].sum())


def test_run_baseline_comparison_missing_dir_raises(tmp_path):
    """Missing data directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_baseline_comparison(str(tmp_path / "nope"))
