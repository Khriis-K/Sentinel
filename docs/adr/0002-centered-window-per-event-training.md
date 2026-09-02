# ADR-0002: Centered-Window Per-Event Training for BiLSTM

**Status:** Accepted
**Date:** 2026-08-09
**Deciders:** Chris, Claude Code (grill session)

## Context

The original training pipeline in `src/train.py` used **sliding windows with window-level labels**: a 512-event window was labeled malicious if it contained at least one `evil==1` event (`label = any(evil_in_window)`). The evaluation concatenated all-benign validation windows with all-malicious test windows and computed AUROC across the combined set.

This had a structural flaw: the model only needed to detect "which dataset did this window come from?" — host identification, not anomaly detection. The all-benign windows came from one set of hosts and the all-malicious windows from the attack host. AUROC was trivially 1.000.

The paper baselines (iForest, Robust Covariance, One-Class SVM) score every individual event in the test CSV — 16% benign, 84% malicious, all from the same attack host. That's a much harder problem, and it's the correct comparison.

> **Dataset scope caveat:** The paper (Highnam et al., 2021) evaluated baselines on the full BETH dataset — 8M+ events across 23 honeypots. Sentinel uses the per-host CSVs from the Kaggle distribution: 2.67M events across 6 hosts (2 attack hosts), split via `split_by_host` into train/val/test. This is larger and more diverse than the pre-split benchmark subset (1.14M events, 1 attack host), but still smaller than the full research dataset. Our reproduced baseline scores will differ from the paper's reported values (0.850, 0.519, 0.605) — the host composition and class distribution are different. The comparison between Sentinel's BiLSTM and the baselines is valid (same data, same features, same per-event evaluation protocol), but absolute numbers should not be directly compared to the paper's.

The fix requires two changes:
1. **Evaluation:** Score every event individually using a 512-event window centered on that event, then compute per-event AUROC.
2. **Training:** Retrain the model with the same centered-window, per-event objective so the model learns exactly the task it's evaluated on.

## Decision

### Shift from window-level to per-event training

The model is retrained with **centered windows labeled by the center event's `evil` value** (`label = evil[i]`), not by `any(evil_in_window)`.

This corrects a label mismatch: under window-level training, a benign event at position `i` surrounded by evil neighbors got a positive label (because the window contained evil). The model learned to fire on any event near an attack, rather than discriminating the attack events themselves. Per-event labeling teaches the model to answer "is *this specific event* anomalous given its temporal context?"

### New `CenteredWindowDataset` class (not a flag on `BethDataset`)

The two dataset classes have fundamentally different contracts:

| Property | `BethDataset` (sliding) | `CenteredWindowDataset` |
|----------|------------------------|--------------------------|
| Window placement | Sequential, stride-based | Centered on each event at position `i` |
| Label | `any(evil_in_window)` | `evil[i]` (center event) |
| Indexing | Window index (0..N_windows) | Event position index in source array |
| Edge handling | N/A (starts at window_size//2 naturally) | Truncate: events without 256 neighbors on both sides are not used as centers |

Retrofitting `BethDataset` with `if centered:` branches would violate the Single Responsibility Principle and make both use cases harder to reason about. A separate class keeps each clean.

### Training window stride: 32

Centered windows are generated with stride=32, producing ~50K training windows from ~1.6M training events (60% of 2.67M). Adjacent windows overlap ~94% (480/512 events shared).

- **Denser (stride=16, ~100K windows):** Burns GPU time on correlated gradients. Adjacent windows share >97% of their events — the marginal information gain per window is negligible.
- **Sparser (stride=128, ~12K windows):** Risks the model never seeing edge cases where the center event's label differs from its neighbors' context. At the boundary between benign and malicious regions of the attack hosts' timelines, stride=32 gives the model multiple windows spanning the transition.
- **Stride=32** balances gradient diversity against coverage: an epoch takes seconds on the RTX 5070, and every event in each attack host's timeline is the center of at least one training window.

### Edge handling: truncate

Events within 256 positions of either end of a host's timeline cannot have a full 512-event window centered on them. These events are simply not used as centers. Losing at most 512 events per host (~3K total out of 2.67M) is negligible, and there is no synthetic data (from padding or mirroring) to confuse the model.

### Class imbalance: keep `train_attack_frac=0.2`

With stride=32, evil-center windows are ~2K out of ~50K total (~25:1 ratio). `BCEWithLogitsLoss` with `pos_weight` handles the numeric imbalance. The subtler risk — benign-center windows in attack hosts sharing temporal context with nearby evil events — is a tuning problem, not an architecture problem. If the model can't disambiguate under these conditions, we can increase `train_attack_frac` or add a contrastive loss component later.

### Mixed-class validation holdout

The existing validation set is all-benign. For the per-event objective, every validation window has a benign center — a model that predicts 0.0 for everything would ace it (the null-case test catches this).

Fix: during the per-host data pipeline (loading CSVs → `split_by_host` → mixing attack data), hold out 20% of the mixed-in attack data as a separate mixed-class validation set. The all-benign validation set stays for loss-based early stopping (its loss is clean and unconfounded by attack data). The mixed validation set provides per-event AUROC for model selection.

### Checkpoint bundling

Save `cat_vocabs`, `numeric_stats`, `process_vocab`, and `args_vocab` in `model.pt` under a `preprocess_artifacts` key. One file, no version skew between model weights and preprocessing state. Inference requires only `model.pt` and a CSV to score.

## Consequences

### Positive
- Model is trained and evaluated on the same task: per-event anomaly scoring
- AUROC comparison against paper baselines is apples-to-apples
- The all-benign val set + mixed val set split provides both a clean loss signal and a meaningful model-selection metric
- Single-file checkpoint simplifies inference and deployment

### Negative
- Existing `outputs/model.pt` (trained with window-level objective) is superseded and must be regenerated
- `CenteredWindowDataset` adds a second dataset class to maintain, though it shares preprocessing with `BethDataset`
- Per-event evaluation is more expensive than window-level evaluation (~50K forward passes), but still runs in seconds on GPU

### Risk: temporal leakage in centered windows
If the test host's timeline contains both benign and malicious regions, a centered window over a benign event may include malicious events in its context. The model must learn to disambiguate the center from its surroundings. This is the right problem to solve — it's exactly what a forensic analyst does when examining an event in context — but it's harder than the window-level task. Mitigation: if the model can't learn this distinction, increase `train_attack_frac` to expose it to more benign-center-in-evil-neighborhood examples during training.

## Alternatives Considered

### A. Keep window-level training, fix only evaluation
Discarded. Training on `any(evil_in_window)` while evaluating on `evil[i]` creates a label mismatch. The model would be optimized for a different objective than the one it's measured on. Early experiments confirmed this: a window-level-trained model assigns high scores to benign events surrounded by evil neighbors, inflating false positives.

### B. Stride=1 (every event is a center)
Discarded. All ~1.6M training events would produce ~1.6M windows. At 94%+ overlap, the gradient signal is highly redundant and training time balloons for no gain. Stride=32 gives effectively the same coverage at 1/32 the cost.

### C. Edge padding (mirror/repeat) instead of truncate
Discarded. Mirroring or repeating boundary events introduces synthetic data that doesn't correspond to any real event sequence. The model could learn artifacts of the padding strategy rather than genuine anomaly patterns. The ~3K lost events are not worth the synthetic-data risk.

### D. Single dataset class with `centered=True` flag
Discarded. The label semantics, indexing scheme, and window construction differ enough between the two modes that a flag-based interface would be confusing and error-prone. KISS: two classes, each doing one thing clearly.
