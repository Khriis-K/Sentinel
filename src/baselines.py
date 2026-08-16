"""
Sentinel — Baseline Models
Reproduces the three anomaly detection baselines from the BETH paper
(Highnam et al., 2021): Isolation Forest, Robust Covariance, One-Class SVM.

Paper reported results (on the 7-feature binarized subset):
  - iForest: 0.850 AUROC
  - Robust Covariance: 0.519 AUROC
  - One-Class SVM: 0.605 AUROC
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from src.data import METRIC_KEYS


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class BaselineResults:
    """Container for baseline evaluation results."""
    iforest: Dict[str, object] = field(default_factory=dict)
    robust_covariance: Dict[str, object] = field(default_factory=dict)
    one_class_svm: Dict[str, object] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a comparison DataFrame."""
        rows = []
        for name, metrics in [
            ("iForest", self.iforest),
            ("Robust Covariance", self.robust_covariance),
            ("One-Class SVM", self.one_class_svm),
        ]:
            row = {"model": name}
            for k in METRIC_KEYS:
                row[k] = metrics.get(k)
            rows.append(row)
        return pd.DataFrame(rows)


# ── Training Functions ─────────────────────────────────────────────────────────

def train_iforest(
    X: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 100,
    seed: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest model on benign training data.

    Args:
        X: Feature array of shape (n_samples, n_features).
        contamination: Expected fraction of anomalies.
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
        contamination: Expected fraction of anomalies.
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


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_baseline(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination: float = 0.01,
) -> Dict[str, object]:
    """Evaluate a fitted anomaly detection model on a held-out test set.

    Uses decision_function: lower scores = more anomalous.
    We invert scores so higher = more anomalous for metric computation.
    Threshold is set using the contamination parameter, not hardcoded.

    Args:
        model: Fitted anomaly detection model with decision_function.
        X_test: Test features (must NOT overlap with training data).
        y_test: Ground truth labels (0 = benign, 1 = malicious).
        contamination: Expected fraction of anomalies (used for threshold).

    Returns:
        Dict with auroc, pr_auc, f1, precision, recall, confusion_matrix.
    """
    raw_scores = model.decision_function(X_test)
    scores = -raw_scores  # invert: higher = more anomalous

    auroc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    # Threshold from contamination: top contamination% → predicted anomaly
    quantile = 1.0 - contamination
    threshold = np.quantile(scores, quantile)
    y_pred = (scores >= threshold).astype(np.int64)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist(),
    }


# ── Run All ────────────────────────────────────────────────────────────────────

def run_all_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination: float = 0.01,
    seed: int = 42,
) -> BaselineResults:
    """Run all three paper baselines with proper train/test separation.

    Models are fitted on X_train (benign only, per BETH split) and
    evaluated on X_test (which contains the attack host). This matches
    the paper's evaluation protocol and prevents label leakage.

    Args:
        X_train: Training features (benign hosts only).
        X_test: Test features (includes attack host).
        y_test: Ground truth labels for test set.
        contamination: Expected anomaly fraction.
        seed: Random seed for iForest and Robust Covariance.

    Returns:
        BaselineResults with metrics for all three models.
    """
    # Fit scaler on training data only, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = BaselineResults()

    # iForest
    iforest = train_iforest(X_train_scaled, contamination=contamination, seed=seed)
    results.iforest = evaluate_baseline(iforest, X_test_scaled, y_test, contamination)

    # Robust Covariance
    robust = train_robust_covariance(X_train_scaled, contamination=contamination, seed=seed)
    results.robust_covariance = evaluate_baseline(robust, X_test_scaled, y_test, contamination)

    # One-Class SVM
    ocsvm = train_one_class_svm(X_train_scaled, nu=contamination)
    results.one_class_svm = evaluate_baseline(ocsvm, X_test_scaled, y_test, contamination)

    return results


# ── Paper-Reproduction Utility ─────────────────────────────────────────────────

def extract_paper_features(df: pd.DataFrame) -> np.ndarray:
    """Extract the 7 binarized features used in the BETH paper baselines.

    The paper used a subset of 7 features, binarized. This replicates
    that preprocessing for direct comparison against reported baselines.

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
