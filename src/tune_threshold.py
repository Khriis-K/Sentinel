"""
Sentinel — Checkpoint Retuning
Load a trained next-event checkpoint, recompute attack-val threshold
tuning, and re-evaluate the test corpus — without retraining. Uses the
same seed-driven pipeline as training, so the attack-val carve-out is
identical to the one used at training time.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_next_event_pipeline
from src.eval import evaluate_scores, json_serialize, tune_threshold
from src.model import NextEventLSTM
from src.train import collate_next_event, collect_scores, get_device


def main(
    checkpoint: str = "outputs/model.pt",
    data_dir: str = "data/raw/per_host",
    output_dir: str = "outputs",
    window_size: int = 512,
    train_stride: int = 8,
    val_stride: int = 16,
    batch_size: int = 256,
    n_blocks: int = 50,
    tune_frac: float = 0.2,
    seed: int = 42,
):
    """Re-tune the decision threshold and re-evaluate test from a checkpoint."""
    device = get_device()
    pin_memory = device.type == "cuda"
    print(f"Device: {device}")

    # ── Load checkpoint ─────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {checkpoint}")
    print(f"  Vocab sizes from checkpoint: {ckpt.get('vocab_sizes', {})}")

    model = NextEventLSTM(**ckpt["init_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Rebuild pipeline (same seed → identical split + carve) ───────────────
    print(f"\nLoading per-host CSVs from {data_dir}...")
    _, _, tune_ds, test_ds, vocab_sizes, _ = load_next_event_pipeline(
        raw_dir=data_dir,
        window_size=window_size,
        train_stride=train_stride,
        val_stride=val_stride,
        n_blocks=n_blocks,
        tune_frac=tune_frac,
        seed=seed,
    )

    tune_loader = DataLoader(
        tune_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_next_event, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_next_event, pin_memory=pin_memory,
    )

    # ── Threshold tuning on attack-val ───────────────────────────────────────
    print("\nScoring attack-val (threshold tuning set)...")
    tune_scores, tune_labels = collect_scores(
        model, tune_loader, device, tune_ds.labels, tune_ds.centers,
    )
    print(f"  {len(tune_scores):,} events, {int(tune_labels.sum()):,} evil")

    print("\nTuning threshold on attack-val (full score range)...")
    threshold, tune_metrics = tune_threshold(tune_scores, tune_labels)
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Attack-val @ threshold: "
          f"F1={tune_metrics['f1']:.4f}, "
          f"P={tune_metrics['precision']:.4f}, "
          f"R={tune_metrics['recall']:.4f}")

    # ── Dense test evaluation ────────────────────────────────────────────────
    print("\nScoring test (dense stride-1, every target event)...")
    test_scores, test_labels = collect_scores(
        model, test_loader, device, test_ds.labels, test_ds.centers,
    )
    print(f"  {len(test_scores):,} events, {int(test_labels.sum()):,} evil")

    metrics = evaluate_scores(test_labels, test_scores, threshold)
    print(f"\nTest @ attack-val threshold:")
    print(f"  AUROC:            {metrics['auroc']:.4f}")
    print(f"  PR-AUC:           {metrics['pr_auc']:.4f}  (primary)")
    print(f"  F1:               {metrics['f1']:.4f}")
    print(f"  Precision@0.1%:   {metrics['precision_at_0_1']:.4f}")
    print(f"  Precision@1%:     {metrics['precision_at_1']:.4f}")
    print(f"  Confusion:        {metrics['confusion_matrix']}")

    # ── Update eval_results.json ─────────────────────────────────────────────
    eval_path = Path(output_dir) / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)
    else:
        eval_results = {}

    eval_results["next_event_lstm"] = metrics
    eval_results["threshold_tuning"] = {
        "threshold": threshold,
        "tune_source": "attack-val (carved from test; test never used for tuning)",
        "tune_size": int(len(tune_scores)),
        "tune_evil": int(tune_labels.sum()),
        "tune_metrics": tune_metrics,
        "method": "maximize F1 over 1000 candidates spanning the full score range",
        "retuned": True,
    }
    eval_results.setdefault("baselines", None)

    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=json_serialize)
    print(f"\nUpdated {eval_path}")


if __name__ == "__main__":
    main()
