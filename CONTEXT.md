# Sentinel

An offline SOC alert triage system that uses an LSTM-VAE (variational autoencoder) to score kernel process events for anomalous (malicious) activity, with LLM-powered incident report generation from detected anomalies. Trains on benign data only; detects anomalies via reconstruction error.

## Language

**Per-event score**:
An anomaly score assigned to a single kernel process event. Computed by centering a 512-event context window around the target event and running that window through the LSTM-VAE. The reconstruction error (MSE between input and output) serves as the anomaly score. This is the evaluation unit that matches the BETH paper baselines.
_Avoid_: Window score, segment score, trace score

**Centered window**:
A 512-event window constructed around a target event at position `i`: 256 events before `i`, 255 events after `i`, plus the event at `i` itself. At CSV edges, pad by mirroring or repeating boundary events.
_Avoid_: Sliding window (that's strided, not centered), context window

**Window-level objective**:
The current training signal: a window is labeled malicious if it contains at least one `evil==1` event (`label = any(evil)`). This conflates host identification with anomaly detection when val is all-benign and test is all-malicious.
_Avoid_: Any-evil label, window label

**Per-event objective**:
The corrected training signal: a centered window is labeled by the center event's `evil` value (`label = evil[i]`). The model learns to answer "is this specific event anomalous?" given its temporal context.
_Avoid_: Event label (ambiguous — could mean the raw CSV column)

**Offline / forensic detection**:
Sentinel's operational mode: the full event log is available at inference time, and scoring happens retrospectively. An SOC analyst investigating an alert pulls the surrounding event window and classifies. Contrasts with streaming detection, where events arrive one at a time and future context is unavailable.
_Avoid_: Post-hoc detection, batch detection

**Host-based split**:
The BETH dataset partitioning strategy: malicious hosts (any evil==1) go to test; benign hosts are split train/val by host, not by row. This ensures the model is evaluated on hosts it has never seen, preventing host-level overfitting.
_Avoid_: Row split, random split
