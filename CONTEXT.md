# Sentinel

An offline SOC alert triage system that uses a next-event LSTM (a DeepLog-lineage language model over kernel process events) to score events for anomalous (malicious) activity, with LLM-powered incident report generation from detected anomalies. Trains on benign data only; scores anomalies by per-event surprisal.

## Language

**Per-event score**:
The anomaly score for a single kernel process event: the sum of per-field surprisals from the next-event LSTM — −log P(field | trailing context), summed over the predicted fields (eventId, processName, userId, returnValue bucket, argsNum bucket). Higher means the model finds the event more improbable given its past. This is the evaluation unit that matches the BETH paper baselines.
_Avoid_: Reconstruction error (the retired VAE objective), window score, segment score, trace score

**Trailing window**:
The 512 events immediately preceding event `i` (positions `i−512 … i−1`) — the context from which the next-event LSTM predicts event `i`. Events near the start of a host log without a full trailing window are skipped.
_Avoid_: Centered window (the retired VAE-era construction), sliding window (that's strided, not trailing), context window

**Offline / forensic detection**:
Sentinel's operational mode: the full event log is available at inference time, and scoring happens retrospectively. An SOC analyst investigating an alert pulls the surrounding event window and classifies. Contrasts with streaming detection, where events arrive one at a time and future context is unavailable.
_Avoid_: Post-hoc detection, batch detection

**Host-based split**:
The BETH dataset partitioning strategy: malicious hosts (any evil==1) go to test; benign hosts are split train/val by host, not by row. This ensures the model is evaluated on hosts it has never seen, preventing host-level overfitting.
_Avoid_: Row split, random split
