"""
Sentinel — Training Loop
Next-event LSTM training (ADR-0004): summed per-field cross-entropy on
benign data only, early stopping and model selection on benign-validation
surprisal, decision threshold tuned on the attack-val carve-out, and
dense stride-1 per-event evaluation on test.
"""
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import TARGET_FIELDS, load_next_event_pipeline
from src.eval import evaluate_scores, tune_threshold
from src.model import NextEventLSTM


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Collation ──────────────────────────────────────────────────────────────────

def collate_next_event(batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Stack a list of (context, targets) tuples into batched tensors.

    Context features stack to (B, window_size, ...); targets stack to (B,).
    """
    contexts, targets = zip(*batch)

    batched_context = {
        key: torch.stack([c[key] for c in contexts]) for key in contexts[0]
    }
    batched_targets = {
        field: torch.stack([t[field] for t in targets]) for field in targets[0]
    }
    return batched_context, batched_targets


# ── Loss ───────────────────────────────────────────────────────────────────────

def next_event_loss(
    logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sum of per-field cross-entropies against the next event's fields.

    No score-weighting parameters (ADR-0004): every field contributes
    equally to the loss.

    Returns:
        (total_loss, per_field_losses) — scalar tensor and dict of scalars.
    """
    per_field = {
        field: F.cross_entropy(logits[field], targets[field])
        for field in TARGET_FIELDS
    }
    total = sum(per_field.values())
    return total, per_field


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(
    model: NextEventLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch. Returns (avg_total_loss, avg_per_field)."""
    model.train()
    total_loss = 0.0
    field_sums = {field: 0.0 for field in TARGET_FIELDS}
    n_batches = 0

    for context, targets in loader:
        context = {k: v.to(device, non_blocking=True) for k, v in context.items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad()
        logits = model(context)
        loss, per_field = next_event_loss(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for field in TARGET_FIELDS:
            field_sums[field] += per_field[field].item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, {field: 0.0 for field in TARGET_FIELDS}
    return (
        total_loss / n_batches,
        {field: field_sums[field] / n_batches for field in TARGET_FIELDS},
    )


@torch.no_grad()
def validate_surprisal(
    model: NextEventLSTM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Average per-event total surprisal over a benign validation set."""
    model.eval()
    surprisal_sum = 0.0
    n_events = 0

    for context, targets in loader:
        context = {k: v.to(device, non_blocking=True) for k, v in context.items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        surprisal = model.surprisal(context, targets)
        surprisal_sum += surprisal["total"].sum().item()
        n_events += surprisal["total"].numel()

    return surprisal_sum / n_events if n_events > 0 else 0.0


@torch.no_grad()
def collect_scores(
    model: NextEventLSTM,
    loader: DataLoader,
    device: torch.device,
    y_true: np.ndarray,
    centers: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-event total surprisal for every scored position in the loader.

    Args:
        model: Trained next-event model.
        loader: DataLoader over a TrailingWindowDataset.
        device: Torch device.
        y_true: Evil labels per event (dataset-level, length n_events).
        centers: The dataset's scored positions (.centers).

    Returns:
        (scores, labels) — total surprisal and evil label per scored event.
    """
    model.eval()
    all_scores = []
    n_batches = len(loader)

    for i, (context, targets) in enumerate(loader):
        context = {k: v.to(device, non_blocking=True) for k, v in context.items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        surprisal = model.surprisal(context, targets)
        all_scores.append(surprisal["total"].cpu())

        if (i + 1) % 500 == 0 or i + 1 == n_batches:
            print(f"    batch {i+1}/{n_batches}", flush=True)

    scores = torch.cat(all_scores).numpy()
    labels = np.asarray(y_true)[np.asarray(centers)].astype(np.int64)
    return scores, labels


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    data_dir: str = "data/raw/per_host",
    output_dir: str = "outputs",
    window_size: int = 512,
    train_stride: int = 8,
    val_stride: int = 16,
    batch_size: int = 64,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    n_blocks: int = 50,
    tune_frac: float = 0.2,
    seed: int = 42,
    evaluate: bool = True,
    run_baselines: bool = True,
):
    """Run the next-event training pipeline.

    Loads per-host CSVs (benign hosts train/val, attack-val carved from
    test), trains the next-event LSTM on summed per-field cross-entropy,
    and saves model.pt. When ``evaluate``, tunes the decision threshold on
    attack-val and evaluates dense stride-1 on test. When
    ``run_baselines``, re-baselines the paper models under the same
    protocol and adds them to eval_results.json.
    """
    # ── Setup ───────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pin_memory = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Output dir: {output_path.resolve()}")

    # ── Load data (benign only for training) ─────────────────────────────────
    print(f"\nLoading per-host CSVs from {data_dir}...")
    train_ds, val_ds, tune_ds, test_ds, vocab_sizes, vocabs = load_next_event_pipeline(
        raw_dir=data_dir,
        window_size=window_size,
        train_stride=train_stride,
        val_stride=val_stride,
        n_blocks=n_blocks,
        tune_frac=tune_frac,
        seed=seed,
    )

    def _describe(name: str, ds) -> None:
        n_evil = int(ds.labels[ds.centers].sum()) if ds.labels is not None and len(ds) else 0
        print(f"  {name:10s} {len(ds):>9,} scored events ({n_evil:,} evil)")

    print("\nSplits (scored target events):")
    _describe("Train", train_ds)
    _describe("Val", val_ds)
    _describe("AttackVal", tune_ds)
    _describe("Test", test_ds)

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_next_event, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_next_event, pin_memory=pin_memory,
    )
    tune_loader = DataLoader(
        tune_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_next_event, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_next_event, pin_memory=pin_memory,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = NextEventLSTM(**vocab_sizes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_surprisal = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print(f"\nTraining ({num_epochs} epochs max, patience={early_stopping_patience}):")
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, per_field = train_epoch(model, train_loader, optimizer, device)
        val_surprisal = validate_surprisal(model, val_loader, device)

        if val_surprisal < best_val_surprisal:
            best_val_surprisal = val_surprisal
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        field_str = " ".join(f"{f[:4]}={per_field[f]:.3f}" for f in TARGET_FIELDS)
        print(
            f"  Epoch {epoch:3d}/{num_epochs} | "
            f"loss: {train_loss:.4f} ({field_str}) | "
            f"val_surprisal: {val_surprisal:.4f}{marker}",
            flush=True,
        )

        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no val-surprisal improvement for {patience_counter} epochs)")
            break

    train_time = time.time() - t_start
    print(f"\nBest val surprisal: {best_val_surprisal:.4f} at epoch {best_epoch}")
    print(f"Training time: {train_time:.0f}s")

    # ── Save checkpoint (before evaluation — a crash keeps the model) ────────
    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = output_path / "model.pt"
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "init_kwargs": model._init_kwargs,
            "vocab_sizes": vocab_sizes,
            "vocabs": vocabs,
        },
        model_path,
    )
    print(f"\nSaved model -> {model_path}")

    eval_results = {
        "next_event_lstm": None,
        "threshold_tuning": None,
        "baselines": None,
        "config": {
            "data_dir": data_dir,
            "window_size": window_size,
            "train_stride": train_stride,
            "val_stride": val_stride,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "best_epoch": best_epoch,
            "best_val_surprisal": best_val_surprisal,
            "train_time_s": train_time,
            "n_blocks": n_blocks,
            "tune_frac": tune_frac,
            "vocab_sizes": vocab_sizes,
            "seed": seed,
            "device": str(device),
            "n_params": n_params,
            "dataset_sizes": {
                "train": len(train_ds),
                "val": len(val_ds),
                "attack_val": len(tune_ds),
                "test": len(test_ds),
            },
        },
    }

    # ── Threshold tuning on attack-val + dense test evaluation ───────────────
    if evaluate:
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

        eval_results["next_event_lstm"] = metrics
        eval_results["threshold_tuning"] = {
            "threshold": threshold,
            "tune_source": "attack-val (carved from test; test never used for tuning)",
            "tune_size": int(len(tune_scores)),
            "tune_evil": int(tune_labels.sum()),
            "tune_metrics": tune_metrics,
            "method": "maximize F1 over 1000 candidates spanning the full score range",
        }

    # ── Paper baselines under the same protocol ──────────────────────────────
    if run_baselines:
        from src.baseline_protocol import run_baseline_comparison

        eval_results["baselines"] = run_baseline_comparison(
            data_dir=data_dir,
            n_blocks=n_blocks,
            tune_frac=tune_frac,
            seed=seed,
        )
        print("\nBaseline comparison (same attack-val protocol):")
        for name, m in eval_results["baselines"].items():
            if isinstance(m, dict) and "auroc" in m:
                print(f"  {name:20s} AUROC={m['auroc']:.4f} PR-AUC={m['pr_auc']:.4f} "
                      f"P@1%={m['precision_at_1']:.4f}")

    # ── Save eval results ────────────────────────────────────────────────────
    eval_path = output_path / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=_json_serialize)
    print(f"Saved eval results -> {eval_path}")

    return eval_results


def _json_serialize(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
