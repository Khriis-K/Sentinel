"""
Sentinel — Standalone Baseline Runner
Re-baselines the paper detectors (iForest, Robust Covariance, One-Class
SVM) under the honest protocol and merges the results into
eval_results.json — the baseline-side counterpart of tune_threshold.py.
Uses the same seed-driven pipeline as the neural model, so the attack-val
carve-out (and its tuned thresholds) refer to identical events.
"""
import json
from pathlib import Path

from src.baseline_protocol import run_baseline_comparison
from src.eval import json_serialize


def main(
    data_dir: str = "data/raw/per_host",
    output_dir: str = "outputs",
    n_blocks: int = 50,
    tune_frac: float = 0.2,
    seed: int = 42,
):
    """Run the baseline comparison and write the 'baselines' results block."""
    print(f"Running baseline comparison (data_dir={data_dir}, seed={seed})...")
    results = run_baseline_comparison(
        data_dir=data_dir,
        n_blocks=n_blocks,
        tune_frac=tune_frac,
        seed=seed,
    )

    eval_path = Path(output_dir) / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)
    else:
        eval_results = {}
    eval_results["baselines"] = results

    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=json_serialize)

    print("\nBaseline comparison (same attack-val protocol):")
    for name, m in results.items():
        if isinstance(m, dict) and "auroc" in m:
            print(f"  {name:20s} AUROC={m['auroc']:.4f} PR-AUC={m['pr_auc']:.4f} "
                  f"P@1%={m['precision_at_1']:.4f}")
    print(f"\nUpdated {eval_path}")


if __name__ == "__main__":
    main()
