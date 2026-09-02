# ADR-0004: Pivot from LSTM-VAE Reconstruction to Next-Event Prediction

**Status:** Accepted (supersedes ADR-0003)
**Date:** 2026-09-01
**Deciders:** Chris, Claude Code

## Context

ADR-0003 pivoted Sentinel from a supervised BiLSTM to an LSTM-VAE, keeping the core principle: train on benign data only, no attack labels. Subsequent investigation (see `docs/handoffs/handoff-model-investigation.md`) found the trained VAE is **fully posterior-collapsed**: the encoder is a constant function of its input (μ std ≈ 0.0000 across 40k windows from different hosts; KL < 1e-3 in all 32 latent dims). The anomaly score degenerated into distance from a single memorized average sequence — explaining AUROC 0.929 alongside PR-AUC 0.045 (vs iForest at PR-AUC 0.257).

The root cause is structural, not a hyperparameter problem:

1. The decoder input is `torch.zeros` (`src/model.py:212`) — the decoder never sees the input, and reconstructing 512×171 dims from one 32-dim latent makes ignoring `z` the training optimum.
2. MSE on mean-pooled embedding vectors is a poor objective for an almost-entirely-categorical event space (39 eventIds, 94 processNames, 9 users).
3. One latent vector must summarize 512 events, capping per-event resolution even if collapse were fixed.

The best achievable scoring variant of the measured architecture reached PR-AUC 0.123 — still half of iForest. ADR-0003's own mitigation line ("if VAE underperforms iForest, something is wrong") fired.

## Decision

**Replace reconstruction with next-event prediction (DeepLog lineage).** One forward LSTM, trained benign-only, reads the trailing 512-event window (events `i−512…i−1`) and predicts event `i` through per-field cross-entropy heads over: `eventId`, `processName`, `userId`, bucketed `returnValue` (error/−1, zero, log1p bins), and bucketed `argsNum`. The anomaly score is the sum of per-field surprisals: −log P(event `i` | trailing context).

ADR-0003's core principle is preserved exactly — benign-only training, no attack labels. `args` content, `mountNamespace`, and `parentProcessId` are excluded from prediction targets (UNK-flooding, constant value, instance-specific noise, respectively) but remain available as input features where applicable.

## Consequences

### Positive
- Posterior collapse is structurally impossible — the prediction heads condition on the actual input.
- The score is per-event and calibrated by construction (log-probabilities), directly answering "how improbable is this event given its past?"
- Simpler failure-mode analysis: one clean question per score ("was P(event | its past) low?").
- Streaming detection becomes available later at zero design cost — the model already uses only past context.

### Negative
- Loses a trace-level latent embedding (the VAE's `z`) that could have fed clustering or the LLM report generator.
- Score behavior inside long attack bursts is harder to interpret: once the trailing context is itself anomalous, surprisal depends on how out-of-distribution context corrupts the hidden state.
- The reconstruction narrative is gone; comparability with the BETH paper holds at the per-event evaluation-unit level only.

### Mitigations
- A regression test retained from the collapse diagnostics: per-field surprisal must vary with the input (the next-event analog of "does z still matter?").
- An eval-hygiene bundle applied to the new model and the iForest baseline alike: chunk-wise attack-val carve-out for threshold tuning (thresholds no longer tuned on test), a fair re-baseline of iForest under the same protocol, precision@0.1%/1%, and a 3-seed host-resplit variance pass gated on the single-split result being competitive.

## Alternatives Considered

### A. Repair the VAE (per-feature CE, teacher forcing, free-bits, cyclical KL annealing, center-position scoring, z=μ at eval)
Rejected. Six interacting fixes against a measured structural ceiling (best scoring variant PR-AUC 0.123); the single-latent bottleneck caps per-event resolution regardless of how well training converges.

### B. Strict DeepLog — predict `eventId` only
Rejected. A 39-way fingerprint is too coarse: a malicious `processName` with a benign `eventId` passes. Much of BETH's discriminating signal lives in the fields we would stop predicting.

### C. Bidirectional centered-context prediction (predict event `i` from past AND future)
Deferred, not rejected. Legitimate in offline/forensic mode (the future is available) and would sharpen P on benign data, but two encoders complicate score debugging inside attack bursts. Revisit as an evidence-driven increment if the trailing model lands close to but below iForest.
