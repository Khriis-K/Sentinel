"""
Sentinel — Baseline Models
The three anomaly detection baselines from the BETH paper
(Highnam et al., 2021): Isolation Forest, Robust Covariance, One-Class SVM.

Baselines contribute RAW anomaly scores only — thresholds are tuned on the
attack-val carve-out with the same procedure as the neural model
(src.eval), never via the paper's contamination constant.

Paper reported results (on the 7-feature binarized subset):
  - iForest: 0.850 AUROC
  - Robust Covariance: 0.519 AUROC
  - One-Class SVM: 0.605 AUROC
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM


# ── Training Functions ─────────────────────────────────────────────────────────

def train_iforest(
    X: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 100,
    seed: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest model on benign training data.

    ``contamination`` only shifts sklearn's decision_function by a constant
    here — ranking and the attack-val-tuned threshold are unaffected.

    Args:
        X: Feature array of shape (n_samples, n_features).
        contamination: Expected fraction of anomalies (sklearn parameter).
        n_estimators: Number of trees in the forest.
        seed: Random seed.

    Returns:
        Fitted IsolationForest model.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def train_robust_covariance(
    X: np.ndarray,
    contamination: float = 0.01,
    seed: int = 42,
) -> EllipticEnvelope:
    """Train a Robust Covariance (Elliptic Envelope) model.

    Args:
        X: Feature array of shape (n_samples, n_features).
        contamination: Expected fraction of anomalies (sklearn parameter).
        seed: Random seed.

    Returns:
        Fitted EllipticEnvelope model.
    """
    model = EllipticEnvelope(
        contamination=contamination,
        random_state=seed,
    )
    model.fit(X)
    return model


def train_one_class_svm(
    X: np.ndarray,
    nu: float = 0.01,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> OneClassSVM:
    """Train a One-Class SVM model.

    Args:
        X: Feature array of shape (n_samples, n_features).
        nu: Upper bound on training errors, lower bound on support vectors.
        kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
        gamma: Kernel coefficient.

    Returns:
        Fitted OneClassSVM model.
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_baseline(model, X: np.ndarray) -> np.ndarray:
    """Raw anomaly scores from a fitted sklearn outlier detector.

    decision_function is inverted so HIGHER = more anomalous, matching the
    next-event surprisal convention. Thresholds are tuned downstream on
    attack-val via src.eval — never set here.

    Args:
        model: Fitted detector with decision_function.
        X: Feature array to score.

    Returns:
        float64 array of anomaly scores (higher = more anomalous).
    """
    return -np.asarray(model.decision_function(X), dtype=np.float64)


# ── Paper-Reproduction Utility ─────────────────────────────────────────────────

def extract_paper_features(df: pd.DataFrame) -> np.ndarray:
    """Extract the 7 binarized features used in the BETH paper baselines.

    The paper used a subset of 7 features, binarized. This replicates
    that preprocessing for direct comparison against reported baselines.
    Extraction is TRANSDUCTIVE (value counts computed over the df it is
    given) — kept for comparability with the paper; the caveat is recorded
    alongside baseline results.

    Paper features (7 binary):
      - processId (unique vs not)
      - parentProcessId (0 vs not)
      - userId (0 vs not)
      - mountNamespace (0 vs not)
      - eventId (unique vs not)
      - argsNum (0 vs not)
      - returnValue (0 vs not)

    Returns:
        (n_samples, 7) float32 array of binarized features.
    """
    n = len(df)
    features = np.zeros((n, 7), dtype=np.float32)

    pid_counts = df["processId"].value_counts()
    features[:, 0] = (df["processId"].map(pid_counts) == 1).astype(np.float32).values

    features[:, 1] = (df["parentProcessId"].fillna(0) == 0).astype(np.float32).values

    features[:, 2] = (df["userId"].fillna(0) == 0).astype(np.float32).values

    if "mountNamespace" in df.columns:
        features[:, 3] = (df["mountNamespace"].fillna(0) == 0).astype(np.float32).values
    # else: column missing from per-host CSVs, leave as zeros

    eid_counts = df["eventId"].value_counts()
    features[:, 4] = (df["eventId"].map(eid_counts) == 1).astype(np.float32).values

    features[:, 5] = (df["argsNum"].fillna(0) == 0).astype(np.float32).values

    features[:, 6] = (df["returnValue"].fillna(0) == 0).astype(np.float32).values

    return features
