"""
Sentinel — Training Loop
LSTM-VAE training with reconstruction loss + KL divergence, early stopping,
and evaluation against the BETH paper baselines.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data import (
    CenteredWindowDataset,
    load_benchmark_splits,
    load_per_host_pipeline,
)
from src.model import SentinelVAE
from src.baselines import (
    BaselineResults,
    evaluate_baseline,
    extract_paper_features,
    train_iforest,
    train_one_class_svm,
    train_robust_covariance,
)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Collation ──────────────────────────────────────────────────────────────────

def collate_fn(batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Stack a list of (features, label) tuples into a batch.

    Each feature in features is stacked along dim 0.
    Labels are stacked and reshaped to (batch_size, 1).
    """
    features_list, labels_list = zip(*batch)

    batched_features = {}
    for key in features_list[0]:
        batched_features[key] = torch.stack([f[key] for f in features_list])

    batched_labels = torch.stack(labels_list).unsqueeze(1)  # (B, 1)
    return batched_features, batched_labels


# ── Loss ───────────────────────────────────────────────────────────────────────

def vae_loss(
    reconstructed: torch.Tensor,
    event_vec: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss = MSE reconstruction + β * KL divergence.

    Args:
        reconstructed: Reconstructed event vectors (B, S, event_dim).
        event_vec: Original event vectors (B, S, event_dim).
        mu: Latent mean (B, latent_dim).
        logvar: Latent log-variance (B, latent_dim).
        beta: Weight for KL divergence term (for annealing).

    Returns:
        (total_loss, recon_loss, kl_loss) each as scalar tensors.
    """
    # Reconstruction loss (MSE)
    recon_loss = ((reconstructed - event_vec) ** 2).mean()

    # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
) -> Tuple[float, float, float]:
    """Run one training epoch. Returns (total_loss, recon_loss, kl_loss)."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for features, _ in loader:
        # Labels not needed for VAE training (benign only)
        features = {k: v.to(device) for k, v in features.items()}

        optimizer.zero_grad()
        reconstructed, mu, logvar = model(features)

        # Compute loss — need event_vec for reconstruction target
        event_vec, _, _ = model.encode(features)
        loss, recon, kl = vae_loss(reconstructed, event_vec, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run one validation epoch. Returns average reconstruction loss."""
    model.eval()
    total_recon = 0.0
    n_batches = 0

    for features, _ in loader:
        features = {k: v.to(device) for k, v in features.items()}

        reconstructed, mu, logvar = model(features)
        event_vec, _, _ = model.encode(features)
        recon = ((reconstructed - event_vec) ** 2).mean()

        total_recon += recon.item()
        n_batches += 1

    return total_recon / n_batches if n_batches > 0 else 0.0


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Compute all evaluation metrics using reconstruction error as anomaly score.

    Returns dict with: pr_auc, auroc, f1, precision, recall, confusion_matrix.
    """
    model.eval()
    all_scores = []
    all_labels = []

    for features, labels in loader:
        features = {k: v.to(device) for k, v in features.items()}

        # Reconstruction error per sample
        event_vec, mu, logvar = model.encode(features)
        z = model.reparameterize(mu, logvar)
        seq_len = event_vec.size(1)
        reconstructed = model.decode(z, seq_len)

        # Per-sample MSE (mean over sequence and feature dims)
        mse = ((reconstructed - event_vec) ** 2).mean(dim=(1, 2))
        all_scores.append(mse.cpu())
        all_labels.append(labels.squeeze(1).cpu())

    scores = torch.cat(all_scores).numpy()
    y_true = torch.cat(all_labels).numpy().astype(np.int64)

    # If all labels are the same, some metrics are undefined
    if len(np.unique(y_true)) < 2:
        return {
            "pr_auc": 0.0,
            "auroc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "confusion_matrix": [[int((y_true == 0).sum()), 0], [int((y_true == 1).sum()), 0]],
        }

    # Threshold at median of scores (tune on validation set in practice)
    threshold = np.median(scores)
    y_pred = (scores >= threshold).astype(np.int64)

    auroc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "pr_auc": pr_auc,
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


# ── Per-Event Evaluation ──────────────────────────────────────────────────────

def evaluate_per_event(
    model: nn.Module,
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    host_lengths: list,
    device: torch.device,
    window_size: int = 512,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Score every valid center event using dense (stride=1) centered windows.

    This is the canonical per-event evaluation that matches the BETH paper
    baseline protocol: every event with sufficient temporal context on both
    sides is scored individually, and AUROC / PR-AUC are computed on the
    resulting per-event score–label pairs.

    Args:
        model: Trained SentinelVAE.
        features: Feature dict from ``preprocess_features``.
        labels: Evil labels (int64 array, same length as features).
        host_lengths: Event counts per contiguous host segment.
        device: Torch device.
        window_size: Events per window (default 512).
        batch_size: Batch size for inference.

    Returns:
        Dict with pr_auc, auroc, f1, precision, recall, confusion_matrix.
    """
    ds = CenteredWindowDataset(
        features, labels, host_lengths,
        window_size=window_size, stride=1,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    return evaluate_model(model, loader, device)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    data_dir: str = "data/raw/per_host",
    output_dir: str = "outputs",
    window_size: int = 512,
    stride: int = 32,
    batch_size: int = 64,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    kl_warmup_epochs: int = 10,
    seed: int = 42,
):
    """Run the full LSTM-VAE training pipeline with per-event centered windows.

    Loads per-host CSVs, trains on benign data only with reconstruction +
    KL divergence loss, evaluates with per-event scoring against the test
    split, runs paper baselines for comparison, and writes model.pt +
    eval_results.json.
    """
    # ── Setup ───────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output dir: {output_path.resolve()}")

    # ── Load data (benign only for training) ─────────────────────────────────
    print(f"\nLoading per-host CSVs from {data_dir}...")
    (
        train_ds, val_ds, test_ds, mixed_val_ds,
        process_vocab, args_vocab, cat_vocabs, numeric_stats, vocab_sizes,
    ) = load_per_host_pipeline(
        raw_dir=data_dir,
        window_size=window_size,
        stride=stride,
        train_attack_frac=0,  # No attack data in training — unsupervised
        seed=seed,
    )
    print(f"  Train:     {len(train_ds)} windows (benign only)")
    print(f"  Val:       {len(val_ds)} windows (all-benign)")
    if mixed_val_ds is not None:
        n_mixed_pos = int(sum(1 for i in range(len(mixed_val_ds)) if mixed_val_ds[i][1].item() == 1))
        print(f"  Mixed Val: {len(mixed_val_ds)} windows ({n_mixed_pos} evil-center)")
    print(f"  Test:      {len(test_ds)} windows")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    mixed_val_loader = None
    if mixed_val_ds is not None and len(mixed_val_ds) > 0:
        # Use stride=1 for mixed-val AUROC monitoring
        mixed_val_dense = CenteredWindowDataset(
            mixed_val_ds.features, mixed_val_ds.labels, mixed_val_ds.host_lengths,
            window_size=window_size, stride=1,
        )
        mixed_val_loader = DataLoader(
            mixed_val_dense, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )

    # ── Model ───────────────────────────────────────────────────────────────
    model_kwargs = {
        "process_name_vocab_size": vocab_sizes["processName"],
        "args_vocab_size": vocab_sizes["args"],
        "user_id_vocab_size": vocab_sizes["userId"],
        "mount_ns_vocab_size": vocab_sizes["mountNamespace"],
        "event_id_vocab_size": vocab_sizes["eventId"],
    }
    model = SentinelVAE(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_mixed_auroc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print(f"\nTraining ({num_epochs} epochs max, patience={early_stopping_patience}, "
          f"KL warmup={kl_warmup_epochs}):")
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        # KL annealing: linear warmup from 0 to 1 over kl_warmup_epochs
        beta = min(1.0, epoch / max(kl_warmup_epochs, 1))

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, beta=beta,
        )
        val_recon = validate_epoch(model, val_loader, device)

        # Early stopping: all-benign val reconstruction loss
        loss_improved = val_recon < best_val_loss
        if loss_improved:
            best_val_loss = val_recon
            patience_counter = 0
        else:
            patience_counter += 1

        # Model selection: mixed-class val AUROC (fallback: val loss)
        mixed_auroc_str = ""
        if mixed_val_loader is not None and len(mixed_val_ds) > 0:
            mixed_metrics = evaluate_model(model, mixed_val_loader, device)
            mixed_auroc = mixed_metrics["auroc"]
            if mixed_auroc > best_mixed_auroc:
                best_mixed_auroc = mixed_auroc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mixed_auroc_str = f" | mixed_auroc: {mixed_auroc:.4f}"
        elif loss_improved:
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        marker = " *" if (mixed_val_loader is not None and mixed_auroc > 0 and mixed_auroc >= best_mixed_auroc) or (mixed_val_loader is None and loss_improved) else ""
        print(
            f"  Epoch {epoch:3d}/{num_epochs} | "
            f"loss: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}) | "
            f"val_recon: {val_recon:.4f} | beta={beta:.2f}"
            f"{mixed_auroc_str}{marker}"
        )

        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (val_recon no improvement for {patience_counter} epochs)")
            break

    train_time = time.time() - t_start
    print(f"\nBest val_recon: {best_val_loss:.4f} at epoch {best_epoch}")
    if mixed_val_loader is not None:
        print(f"Best mixed-val AUROC: {best_mixed_auroc:.4f}")
    print(f"Training time: {train_time:.0f}s")

    # ── Load best model ─────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Per-event evaluation on test split ──────────────────────────────────
    print("\nPer-event evaluation on test split (stride=1, every center event)...")
    test_labels = test_ds.labels
    test_host_lengths = test_ds.host_lengths
    bilstm_metrics = evaluate_per_event(
        model, test_ds.features, test_labels, test_host_lengths,
        device, window_size=window_size, batch_size=batch_size * 4,
    )

    print(f"  PR-AUC:     {bilstm_metrics['pr_auc']:.4f}  (primary)")
    print(f"  AUROC:      {bilstm_metrics['auroc']:.4f}")
    print(f"  F1:         {bilstm_metrics['f1']:.4f}")
    print(f"  Precision:  {bilstm_metrics['precision']:.4f}")
    print(f"  Recall:     {bilstm_metrics['recall']:.4f}")
    print(f"  Confusion:  {bilstm_metrics['confusion_matrix']}")

    # ── Run paper baselines on per-host test data ───────────────────────────
    print("\nRunning paper baselines on per-host test data...")

    # Reconstruct per-host data for sklearn baselines
    import src.data as data_module
    host_dfs = data_module.load_beth_data(data_dir)
    sorted_parts = []
    for host in sorted(host_dfs):
        df = host_dfs[host]
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        sorted_parts.append(df)
    full_df_bl = pd.concat(sorted_parts, ignore_index=True)
    train_df_bl, _, test_df_bl = data_module.split_by_host(full_df_bl, seed=seed)

    rng_bl = np.random.default_rng(seed)
    n_bl_train = min(10000, len(train_df_bl))
    train_sample = train_df_bl.iloc[rng_bl.choice(len(train_df_bl), n_bl_train, replace=False)]

    X_train_paper = extract_paper_features(train_sample)
    X_test_paper = extract_paper_features(test_df_bl)
    y_test_paper = test_df_bl["evil"].values.astype(np.int64)

    baseline_results = BaselineResults()

    # iForest
    iforest = train_iforest(X_train_paper, seed=seed)
    baseline_results.iforest = evaluate_baseline(iforest, X_test_paper, y_test_paper)

    # Robust Covariance
    try:
        robust = train_robust_covariance(X_train_paper, seed=seed)
        baseline_results.robust_covariance = evaluate_baseline(robust, X_test_paper, y_test_paper)
    except Exception as e:
        print(f"  Robust Covariance skipped: {e}")

    # One-Class SVM
    try:
        n_svm = min(3000, n_bl_train)
        X_svm = X_train_paper[:n_svm]
        ocsvm = train_one_class_svm(X_svm)
        baseline_results.one_class_svm = evaluate_baseline(ocsvm, X_test_paper, y_test_paper)
    except Exception as e:
        print(f"  One-Class SVM skipped: {e}")

    for name, metrics in [
        ("iForest", baseline_results.iforest),
        ("Robust Covariance", baseline_results.robust_covariance),
        ("One-Class SVM", baseline_results.one_class_svm),
    ]:
        if metrics is not None:
            print(f"  {name:20s}: AUROC={metrics['auroc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")

    # ── Build comparison table ──────────────────────────────────────────────
    comparison = baseline_results.to_dataframe()
    vae_row = pd.DataFrame([{
        "model": "LSTM-VAE (Sentinel)",
        "auroc": bilstm_metrics["auroc"],
        "pr_auc": bilstm_metrics["pr_auc"],
        "f1": bilstm_metrics["f1"],
        "precision": bilstm_metrics["precision"],
        "recall": bilstm_metrics["recall"],
    }])
    comparison = pd.concat([vae_row, comparison], ignore_index=True)

    print(f"\n{'='*70}")
    print("Comparison: LSTM-VAE vs. Paper Baselines")
    print(f"{'='*70}")
    print(comparison.to_string(index=False))

    # ── Save artifacts ──────────────────────────────────────────────────────
    model_path = output_path / "model.pt"
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "init_kwargs": model._init_kwargs,
            "vocab_sizes": vocab_sizes,
            "preprocess_artifacts": {
                "process_vocab": process_vocab,
                "args_vocab": args_vocab,
                "cat_vocabs": cat_vocabs,
                "numeric_stats": numeric_stats,
            },
        },
        model_path,
    )
    print(f"\nSaved model -> {model_path}")

    # Save eval results
    eval_results = {
        "vae": bilstm_metrics,
        "baselines": {
            "iforest": baseline_results.iforest,
            "robust_covariance": baseline_results.robust_covariance,
            "one_class_svm": baseline_results.one_class_svm,
        },
        "comparison": comparison.to_dict(orient="records"),
        "config": {
            "window_size": window_size,
            "stride": stride,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_mixed_auroc": best_mixed_auroc if mixed_val_loader is not None else None,
            "kl_warmup_epochs": kl_warmup_epochs,
            "train_time_s": train_time,
            "vocab_sizes": vocab_sizes,
            "device": str(device),
        },
    }
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
