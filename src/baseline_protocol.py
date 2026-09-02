"""
Sentinel — Baseline Protocol
Re-baselines the BETH paper's anomaly detectors under the honest
evaluation protocol (ADR-0004): raw scores, thresholds tuned on the
attack-val carve-out exactly like the neural model, precision@k added.
Test is never used for tuning.
"""
from typing import Dict

import numpy as np

from src.baselines import (
    extract_paper_features,
    score_baseline,
    train_iforest,
    train_one_class_svm,
    train_robust_covariance,
)
from src.data import load_host_splits
from src.eval import evaluate_scores, tune_threshold

BASELINE_CAVEAT = (
    "Paper baselines keep the BETH paper's transductive feature extraction "
    "(value-count binarization computed over each evaluation corpus); "
    "attack-val and test features are computed on their own corpora. "
    "Thresholds are tuned on attack-val, not via the paper's contamination "
    "constant."
)


def run_baseline_comparison(
    data_dir: str,
    train_sample_size: int = 10000,
    svm_sample_size: int = 3000,
    n_blocks: int = 50,
    tune_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, object]:
    """Score the paper baselines under the same protocol as the neural model.

    For each baseline: train on a benign-only sample (paper's transductive
    7-feature extraction), score the attack-val and test corpora, tune the
    decision threshold on attack-val (full-range F1 maximization — the
    same src.eval procedure the next-event model uses), then evaluate the
    test corpus at that threshold with the full metric set including
    precision@0.1%/1%.

    Args:
        data_dir: Directory containing per-host BETH CSVs.
        train_sample_size: Cap on benign training rows (compute bound).
        svm_sample_size: Cap on training rows for the One-Class SVM.
        n_blocks: Contiguous blocks per test host in the carve-out.
        tune_frac: Probability each carved block lands in the tuning set.
        seed: Random seed (must match the neural pipeline for identical
            split + carve).

    Returns:
        Dict of baseline name → metrics dict (plus 'caveat').
    """
    # Same seed as the neural pipeline → identical split + carve, so every
    # model is thresholded on the same tuning events.
    train_df, _, tune_df, test_df = load_host_splits(
        data_dir, n_blocks=n_blocks, tune_frac=tune_frac, seed=seed,
    )

    rng = np.random.default_rng(seed)
    n_train = min(train_sample_size, len(train_df))
    train_sample = train_df.iloc[rng.choice(len(train_df), n_train, replace=False)]

    X_train = extract_paper_features(train_sample)
    X_tune = extract_paper_features(tune_df)
    X_test = extract_paper_features(test_df)
    y_tune = tune_df["evil"].values.astype(np.int64) if "evil" in tune_df.columns \
        else np.zeros(len(tune_df), dtype=np.int64)
    y_test = test_df["evil"].values.astype(np.int64) if "evil" in test_df.columns \
        else np.zeros(len(test_df), dtype=np.int64)

    results: Dict[str, object] = {}

    baselines = {
        "iforest": lambda: train_iforest(X_train, seed=seed),
        "robust_covariance": lambda: train_robust_covariance(X_train, seed=seed),
        "one_class_svm": lambda: train_one_class_svm(
            X_train[: min(svm_sample_size, len(X_train))]
        ),
    }

    for name, train_fn in baselines.items():
        model = train_fn()
        tune_scores = score_baseline(model, X_tune)
        threshold, tune_metrics = tune_threshold(tune_scores, y_tune)
        test_scores = score_baseline(model, X_test)
        metrics = evaluate_scores(y_test, test_scores, threshold)
        metrics["threshold_tuning"] = {
            "threshold": threshold,
            "tune_source": "attack-val",
            "tune_metrics": tune_metrics,
        }
        results[name] = metrics

    results["caveat"] = BASELINE_CAVEAT
    return results
