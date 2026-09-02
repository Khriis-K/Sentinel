# ADR-0003: Pivot from BiLSTM to LSTM-VAE for Anomaly Detection

**Status:** Accepted
**Date:** 2026-08-15
**Deciders:** Chris, Claude Code

## Context

Sentinel was originally designed around a supervised BiLSTM classifier (ADR-0001). The BiLSTM processes 512-event centered windows and outputs a binary logit — is this event malicious or not?

This created a fundamental tension: the BETH dataset and paper protocol frame the problem as **unsupervised anomaly detection** (train on benign data, detect unseen attacks), but the BiLSTM is a **supervised classifier** that needs labeled attack examples. The workaround was to mix 20% of attack-host events into training, but this produced:

1. **Suspicious results**: AUROC = 1.0 on the test set, suggesting data leakage or overfitting
2. **Extreme class imbalance**: Only 51 evil windows out of 29K training windows (stride=32)
3. **Framing mismatch**: Mixing attack data into training undermines the anomaly detection narrative — the model is learning attack signatures, not detecting novel deviations

The core issue is architectural: a supervised classifier is the wrong tool for unsupervised anomaly detection.

## Decision

**Pivot to an LSTM-VAE (Variational Autoencoder with LSTM encoder/decoder).**

The LSTM-VAE:
- **Trains on benign data only** — no attack labels needed, matching the BETH paper protocol
- **Detects anomalies via reconstruction error** — events the model hasn't seen before reconstruct poorly
- **Uses a probabilistic latent space** — μ and log σ² from the encoder, sampled via the reparameterization trick
- **Learns temporal patterns** — LSTM layers capture sequential structure in process events (unlike iForest/RobustCov which are per-event)
- **Provides principled anomaly scoring** — reconstruction error + KL divergence regularization

The embedding stack (processName, args, userId, mountNamespace, eventId + numeric features) is preserved verbatim — it's architecture-agnostic.

## Architecture

```
Input (512 events × 171-dim event vectors)
  ↓
Encoder LSTM (event_dim → hidden → μ, log σ²)
  ↓
Reparameterization: z = μ + σ * ε
  ↓
Decoder LSTM (latent_dim → hidden → reconstructed events)
  ↓
Output (512 events × 171-dim reconstructed vectors)

Loss = MSE reconstruction + β * KL(q(z|x) || p(z))
Anomaly score = per-window MSE
```

## Consequences

### Positive
- Architecture matches the anomaly detection problem framing
- No attack data in training — cleaner data pipeline, no leakage risk
- Baselines (iForest, Robust Covariance, One-Class SVM) are already unsupervised — fair comparison
- Reconstruction error is interpretable ("how different is this event from normal?")
- LSTM layers still capture temporal patterns in process sequences

### Negative
- Loses the ability to learn specific attack signatures (but that's the point — we want generalization)
- VAE training requires tuning KL weight (β) — may need KL annealing for stable training
- Reconstruction-based scoring can miss subtle anomalies that don't affect reconstruction quality

### Mitigations
- KL annealing (β warmup) prevents posterior collapse during early training
- Per-event evaluation protocol (ADR-0002) still applies — dense stride=1 centered windows
- Baseline comparison provides a sanity check — if VAE underperforms iForest, something is wrong

## Alternatives Considered

### A. Keep BiLSTM, increase attack data mixing
Rejected. More attack data in training moves the problem from anomaly detection to supervised classification. The AUROC=1.0 result suggests the model is already overfitting to the mixed-in attack patterns.

### B. MLP Autoencoder (flatten + compress)
Rejected. Loses temporal structure in the 512-event sequence. Process events are inherently sequential — a malicious process spawning children looks very different from benign sequential activity. An MLP treats the window as a bag of features.

### C. Plain LSTM-AE (no variational component)
Considered. Simpler, but the variational latent space acts as a regularizer that prevents the model from learning an identity function (trivial reconstruction). The KL divergence encourages a smooth, structured latent space that generalizes better to unseen patterns.
