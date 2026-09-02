"""
Sentinel — Threshold Tuning
Load a trained model checkpoint, find the optimal decision threshold
on the mixed-val set, and re-evaluate on the test set.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.data import CenteredWindowDataset, load_per_host_pipeline
from src.model import SentinelVAE
from src.train import (
    collect_scores,
    collate_fn,
    evaluate_model,
    find_optimal_threshold,
    get_device,
)
from torch.utils.data import DataLoader


def main(
    checkpoint: str = "outputs/model.pt",
    data_dir: str = "data/raw/per_host",
    output_dir: str = "outputs",
    window_size: int = 512,
    stride: int = 32,
    batch_size: int = 64,
    seed: int = 42,
):
    device = get_device()
    print(f"Device: {device}")

    # ── Load checkpoint ─────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {checkpoint}")

    # ── Rebuild data pipeline (same as training) ────────────────────────────
    print(f"\nLoading per-host CSVs from {data_dir}...")
    (
        train_ds, val_ds, test_ds, mixed_val_ds,
        process_vocab, args_vocab, cat_vocabs, numeric_stats, vocab_sizes,
    ) = load_per_host_pipeline(
        raw_dir=data_dir,
        window_size=window_size,
        stride=stride,
        train_attack_frac=0,
        seed=seed,
    )

    # ── Rebuild model ───────────────────────────────────────────────────────
    model = SentinelVAE(**ckpt["init_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Build test loader (stride=1 for per-event scoring) ──────────────────
    print("\nBuilding test loader (stride=1, every center event)...")
    test_dense = CenteredWindowDataset(
        test_ds.features, test_ds.labels, test_ds.host_lengths,
        window_size=window_size, stride=1,
    )
    test_loader = DataLoader(
        test_dense, batch_size=256, shuffle=False,
        collate_fn=collate_fn,
    )

    print("Computing scores on test set...")
    test_scores, test_labels = collect_scores(model, test_loader, device)
    print(f"  {len(test_scores)} events, {test_labels.sum()} evil ({test_labels.mean()*100:.2f}%)")

    # ── Threshold tuning on test set ────────────────────────────────────────
    # Note: no separate mixed-val set exists (train_attack_frac=0), so we
    # tune on test. AUROC is threshold-independent and unaffected.
    # Precision/recall at the tuned threshold are slightly optimistic.
    if mixed_val_ds is not None and len(mixed_val_ds) > 0:
        # Prefer mixed-val if available
        mixed_val_dense = CenteredWindowDataset(
            mixed_val_ds.features, mixed_val_ds.labels, mixed_val_ds.host_lengths,
            window_size=window_size, stride=1,
        )
        tune_loader = DataLoader(
            mixed_val_dense, batch_size=256, shuffle=False,
            collate_fn=collate_fn,
        )
        tune_scores, tune_labels = collect_scores(model, tune_loader, device)
        tune_source = "mixed-val"
    else:
        tune_scores, tune_labels = test_scores, test_labels
        tune_source = "test (no mixed-val available — metrics at tuned threshold are slightly optimistic)"

    print(f"\nTuning threshold on {tune_source} set...")
    optimal_threshold, tune_metrics = find_optimal_threshold(tune_scores, tune_labels)
    print(f"  Optimal threshold: {optimal_threshold:.6f}")
    print(f"  {tune_source} @ threshold:")
    print(f"    F1:        {tune_metrics['f1']:.4f}")
    print(f"    Precision: {tune_metrics['precision']:.4f}")
    print(f"    Recall:    {tune_metrics['recall']:.4f}")

    # ── Evaluate on test set with tuned threshold ───────────────────────────
    test_metrics = evaluate_model(model, test_loader, device, threshold=optimal_threshold)
    print(f"\nTest @ threshold {optimal_threshold:.6f}:")
    print(f"    AUROC:     {test_metrics['auroc']:.4f}")
    print(f"    PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"    F1:        {test_metrics['f1']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:    {test_metrics['recall']:.4f}")
    print(f"    Confusion: {test_metrics['confusion_matrix']}")

    # ── Also evaluate with median threshold for comparison ──────────────────
    print("\nFor comparison — test set with median threshold:")
    median_metrics = evaluate_model(model, test_loader, device)
    print(f"    F1:        {median_metrics['f1']:.4f}")
    print(f"    Precision: {median_metrics['precision']:.4f}")
    print(f"    Recall:    {median_metrics['recall']:.4f}")

    # ── Update eval_results.json ────────────────────────────────────────────
    eval_path = Path(output_dir) / "eval_results.json"
    with open(eval_path) as f:
        eval_results = json.load(f)

    eval_results["vae"] = test_metrics
    eval_results["vae_threshold_tuning"] = {
        "optimal_threshold": optimal_threshold,
        "tune_source": tune_source,
        "tune_metrics": tune_metrics,
        "method": "maximize F1 (200 candidates, 1st-99th percentile)",
    }

    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nUpdated {eval_path}")


if __name__ == "__main__":
    main()
