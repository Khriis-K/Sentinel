# ADR-0001: Bidirectional LSTM for Offline Forensic Anomaly Detection

**Status:** Superseded by ADR-0003
**Date:** 2026-08-09
**Deciders:** Chris, Claude Code (grill session)

## Context

Sentinel uses a bidirectional LSTM to score kernel process events for anomaly detection. The architecture processes a 512-event window centered on a target event, with the BiLSTM attending to both past and future context relative to that center.

This raises a natural question: **is bidirectional context cheating?** In streaming/real-time anomaly detection, a model that peeks at future events is clearly illegitimate — those events haven't happened yet. But Sentinel does not operate in streaming mode.

Sentinel's operational mode is **offline forensic detection**: the full event log is available at inference time, and scoring happens retrospectively. An SOC analyst investigating an alert pulls the surrounding event window from a complete log and classifies. The system never needs to make a prediction before the next event arrives.

The distinction matters because a unidirectional LSTM would discard half the available signal: anomalous events often manifest as deviations from the surrounding pattern in both directions. A benign `curl` invocation looks suspicious in isolation but is exonerated by the `apt-get install` that follows it. Conversely, a malicious process may look identical to normal activity when viewed only through its predecessors, but the child processes it spawns reveal the attack.

## Decision

**Use a bidirectional LSTM.** The model attends to both past and future context within each 512-event window. This is legitimate because:

1. **The use case is offline/forensic, not streaming.** The full event log is available at inference time. There is no future-peeking problem because there is no real-time constraint.
2. **Bidirectional context is standard in log forensics.** Malware call-trace analysis, APT hunting, and post-breach investigation all assume retrospective access to the full timeline. Sentinel fits this paradigm, not the streaming-alert paradigm.
3. **The paper baselines also see the full feature vector.** iForest, Robust Covariance, and One-Class SVM don't use temporal context at all — they operate on per-event features extracted after the fact. A unidirectional LSTM would be artificially handicapped relative to these baselines, not fairly matched.

> **Dataset scope caveat:** The paper (Highnam et al., 2021) evaluated baselines on the full BETH dataset — 8M+ events across 23 honeypots. Sentinel uses the pre-split benchmark subset (763K training / 189K validation / 189K test events, with a handful of hosts). Baseline AUROC scores reproduced on this subset differ from the paper's reported values (e.g., iForest 0.827 here vs. 0.850 in the paper). This is expected — the subset is smaller, less diverse, and the attack host represents a larger fraction of the test set. The comparison is still valid (same data, same features, same evaluation protocol), but the absolute numbers are not directly comparable to the paper's.

## Consequences

### Positive
- Model captures deviations from normal patterns in both temporal directions
- Architecture matches the forensic use case described in the PRD
- No need to redesign the model or training pipeline

### Negative
- Sentinel cannot be dropped into a streaming SIEM pipeline without architectural changes (a unidirectional variant would be needed for that use case)
- The distinction between offline vs. streaming must be documented clearly so readers don't mistake this for an evaluation error

### Mitigations
- The `CONTEXT.md` glossary defines **offline / forensic detection** explicitly
- The PRD's problem statement and dashboard design make clear this is retrospective triage, not real-time alerting

## Alternatives Considered

### A. Unidirectional LSTM
Discarded. Would discard half the context signal with no benefit — Sentinel has no streaming requirement. A unidirectional model could be revisited as a variant if a streaming deployment is ever needed, but that's not the current use case.

### B. Transformer (self-attention over sequence)
Discarded as YAGNI. Transformers are more parameter-hungry and data-hungry. For ~24K training windows of 512 events each, a 1.35M-parameter BiLSTM is appropriately scaled. If per-event BiLSTM underperforms the paper baselines after correct evaluation, a Transformer variant becomes worth exploring.

### C. Feed-forward on aggregated event features
Discarded. Aggregating 512 events into a single feature vector via mean/max pooling discards the sequential structure that distinguishes attack patterns from normal variation. The BETH attack (botnet node installation) involves a specific temporal sequence of process creations, not just a statistical shift in feature distributions.
