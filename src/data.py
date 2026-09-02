"""
Sentinel — Data Pipeline
BETH dataset loading, host-based splitting, categorical vocabularies,
deterministic bucketizers, trailing-window next-event datasets, the
attack-val carve-out, and the end-to-end pipeline (ADR-0004).
"""
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Constants ──────────────────────────────────────────────────────────────────

BETH_COLUMNS = [
    "timestamp", "processId", "threadId", "parentProcessId",
    "userId", "mountNamespace", "processName", "hostName",
    "eventId", "eventName", "argsNum", "returnValue",
    "stackAddresses", "args",
]

METRIC_KEYS = ["auroc", "pr_auc", "f1", "precision", "recall"]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_beth_data(raw_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from a directory, keyed by filename stem.

    Args:
        raw_dir: Path to directory containing BETH CSV files.

    Returns:
        Dict of filename stem → DataFrame for each CSV found.
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        return {}

    dfs = {}
    for csv_file in raw_path.glob("*.csv"):
        hostname = csv_file.stem  # filename without .csv
        try:
            df = pd.read_csv(csv_file)
            dfs[hostname] = df
        except Exception as e:
            warnings.warn(f"Skipping unparseable CSV {csv_file.name}: {e}")

    return dfs


# ── Host-Based Splitting ───────────────────────────────────────────────────────

def split_by_host(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val/test by host, not by row.

    Malicious hosts (any evil==1 row) are placed in the test set,
    matching the BETH paper's setup where training contains only benign data.
    Remaining benign hosts fill the rest of the test set after train/val.

    Args:
        df: Full DataFrame with 'hostName' and optional 'evil' column.
        train_frac: Fraction of benign hosts for training.
        val_frac: Fraction of benign hosts for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) tuples.
    """
    rng = np.random.default_rng(seed)
    hosts = df["hostName"].unique()

    # Separate evil and benign hosts
    if "evil" in df.columns:
        evil_hosts = set(df[df["evil"] == 1]["hostName"].unique())
    else:
        evil_hosts = set()

    benign_hosts = [h for h in hosts if h not in evil_hosts]
    evil_hosts_list = list(evil_hosts)

    # Shuffle benign hosts
    rng.shuffle(benign_hosts)

    # Split benign hosts
    n_benign = len(benign_hosts)
    n_train = max(1, int(n_benign * train_frac))
    n_val = max(1, int(n_benign * val_frac))

    train_hosts = benign_hosts[:n_train]
    val_hosts = benign_hosts[n_train:n_train + n_val]
    # Test gets remaining benign hosts + all evil hosts
    test_hosts = benign_hosts[n_train + n_val:] + evil_hosts_list

    # Build splits
    train_df = df[df["hostName"].isin(train_hosts)].copy()
    val_df = df[df["hostName"].isin(val_hosts)].copy()
    test_df = df[df["hostName"].isin(test_hosts)].copy()

    return train_df, val_df, test_df


# ── Labels ─────────────────────────────────────────────────────────────────────

def _get_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract evil labels from a DataFrame, defaulting to zeros if absent."""
    if "evil" in df.columns:
        return df["evil"].values.astype(np.int64)
    return np.zeros(len(df), dtype=np.int64)


# ── Categorical Vocabulary ─────────────────────────────────────────────────────

def build_categorical_vocab(series: pd.Series, max_size: int = 50000) -> Dict[int, int]:
    """Build a value→index mapping for categorical integer features.

    Index 0 is reserved for out-of-vocabulary (unseen) values and is never
    assigned to a real value — real value 0 (e.g. root userId) gets its own
    index like any other value. Real values are assigned indices 1..n in
    descending frequency order, ties broken by ascending value so the mapping
    is deterministic.

    Args:
        series: Integer feature values.
        max_size: Maximum number of unique real values to keep (most frequent).

    Returns:
        Dict mapping original integer value → index (indices start at 1).
    """
    value_counts = series.fillna(0).astype(np.int64).value_counts()
    # Sort by frequency desc, then value asc — value_counts alone does not
    # guarantee tie order, which would make the vocab non-deterministic.
    ordered = sorted(
        ((int(val), int(count)) for val, count in value_counts.items()),
        key=lambda vc: (-vc[1], vc[0]),
    )

    vocab: Dict[int, int] = {}
    for val, _ in ordered:
        if len(vocab) >= max_size:
            break
        vocab[val] = len(vocab) + 1  # 0 reserved for OOV

    return vocab


def map_categorical(series: pd.Series, vocab: Dict[int, int]) -> np.ndarray:
    """Map a Series of integer values through a categorical vocabulary.

    Values not in the vocab map to the reserved OOV index 0.

    Args:
        series: Integer feature values.
        vocab: Value → index mapping from build_categorical_vocab.

    Returns:
        int64 numpy array of mapped indices.
    """
    return np.array(
        [vocab.get(int(v), 0) for v in series.fillna(0)],
        dtype=np.int64,
    )


def build_process_name_vocab(series: pd.Series, max_size: int = 50000) -> Dict[str, int]:
    """Build an exact-string value→index mapping for processName.

    Index 0 is reserved for OOV (unseen names). Names are assigned indices
    1..n in descending frequency order, ties broken alphabetically so the
    mapping is deterministic.

    Args:
        series: Raw processName strings.
        max_size: Maximum number of unique names to keep (most frequent).

    Returns:
        Dict mapping processName string → index (indices start at 1).
    """
    counts = series.fillna("").astype(str).value_counts()
    ordered = sorted(counts.items(), key=lambda nc: (-nc[1], nc[0]))

    vocab: Dict[str, int] = {}
    for name, _ in ordered:
        if len(vocab) >= max_size:
            break
        vocab[name] = len(vocab) + 1  # 0 reserved for OOV

    return vocab


def map_process_name(series: pd.Series, vocab: Dict[str, int]) -> np.ndarray:
    """Map processName strings through the exact-string vocabulary.

    Names not in the vocab map to the reserved OOV index 0.

    Args:
        series: Raw processName strings.
        vocab: Name → index mapping from build_process_name_vocab.

    Returns:
        int64 numpy array of mapped indices.
    """
    return np.array(
        [vocab.get(str(v), 0) for v in series.fillna("")],
        dtype=np.int64,
    )


# ── Numeric Bucketizers ────────────────────────────────────────────────────────
#
# Deterministic, not fitted from data: train/test schemas match by construction.

RETURN_VALUE_CLASSES = 13
ARGS_NUM_CLASSES = 16


def bucket_return_value(series: pd.Series) -> np.ndarray:
    """Bucket returnValue into 13 deterministic classes.

    Class 0: error (any negative value, −1 in practice).
    Class 1: zero (success).
    Classes 2..12: positive magnitudes, 2 + min(ceil(log1p(v)), 10).
    Class 2 is unreachable for integer inputs (ceil(log1p(v)) >= 1 for
    v >= 1); it is reserved so the schema spans exactly 13 indices.

    Args:
        series: Raw returnValue values.

    Returns:
        int64 array of bucket classes in [0, 12].
    """
    v = series.fillna(0).astype(np.int64).values
    out = np.ones(len(v), dtype=np.int64)  # default: zero class
    out[v < 0] = 0
    pos = v > 0
    if pos.any():
        out[pos] = 2 + np.minimum(
            np.ceil(np.log1p(v[pos])).astype(np.int64), 10
        )
    return out


def bucket_args_num(series: pd.Series) -> np.ndarray:
    """Bucket argsNum into 16 deterministic classes (count capped at 15+).

    Args:
        series: Raw argsNum values.

    Returns:
        int64 array of bucket classes in [0, 15].
    """
    v = series.fillna(0).astype(np.int64).values
    return np.clip(v, 0, ARGS_NUM_CLASSES - 1).astype(np.int64)


# ── Attack-Val Carve-Out ───────────────────────────────────────────────────────

def chunk_contiguous_blocks(n: int, n_blocks: int) -> List[Tuple[int, int]]:
    """Split [0, n) into contiguous (start, end) blocks tiling the range.

    Blocks are size-balanced and never overlap; their union is exactly
    [0, n). When n < n_blocks, degrades to n single-event blocks.

    Args:
        n: Total number of positions.
        n_blocks: Requested number of blocks.

    Returns:
        List of (start, end) tuples in ascending order.
    """
    n_blocks = max(1, min(n_blocks, n))
    boundaries = np.linspace(0, n, n_blocks + 1).astype(np.int64)
    return [
        (int(boundaries[i]), int(boundaries[i + 1]))
        for i in range(n_blocks)
        if boundaries[i] < boundaries[i + 1]
    ]


def carve_attack_val(
    test_df: pd.DataFrame,
    n_blocks: int = 50,
    tune_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a threshold-tuning set out of the host-based test split.

    Each test host's time-ordered event stream is cut into contiguous
    blocks; a seeded draw assigns each block to the tuning set or leaves it
    in test. Carved events are removed from test, so tune ∩ test = ∅ and
    tune ∪ test = original. Blocks are atomic — a block is never split.

    Guarantees (via fix-up): the tuning set contains evil events from every
    evil test host and benign events from every benign test host, even when
    the seeded draw misses them (e.g. an attack burst inside one block).

    Args:
        test_df: Test-split DataFrame (must contain 'hostName'; 'evil' and
            'timestamp' used when present).
        n_blocks: Contiguous blocks per host.
        tune_frac: Probability each block is drawn into the tuning set.
        seed: Random seed for block assignment.

    Returns:
        (tune_df, test_df) with carved rows removed from test.
    """
    rng = np.random.default_rng(seed)
    tune_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for host in sorted(test_df["hostName"].unique()):
        host_df = test_df[test_df["hostName"] == host].copy()
        if "timestamp" in host_df.columns:
            host_df.sort_values("timestamp", kind="mergesort", inplace=True)
        host_df.reset_index(drop=True, inplace=True)

        blocks = chunk_contiguous_blocks(len(host_df), n_blocks)
        evil_mask = (
            (host_df["evil"] == 1).values
            if "evil" in host_df.columns
            else np.zeros(len(host_df), dtype=bool)
        )
        is_evil_host = bool(evil_mask.any())

        assigned = rng.random(len(blocks)) < tune_frac

        # Fix-up: guarantee evil coverage from this evil host in tune.
        if is_evil_host:
            has_evil_in_tune = any(
                assigned[k] and evil_mask[s:e].any()
                for k, (s, e) in enumerate(blocks)
            )
            if not has_evil_in_tune:
                for k, (s, e) in enumerate(blocks):  # time order — deterministic
                    if evil_mask[s:e].any():
                        assigned[k] = True
                        break

        # Fix-up: guarantee benign-host representation in tune.
        if not is_evil_host and not assigned.any():
            assigned[0] = True

        for k, (s, e) in enumerate(blocks):
            part = host_df.iloc[s:e]
            (tune_parts if assigned[k] else test_parts).append(part)

    empty = test_df.iloc[0:0].copy()
    tune_df = pd.concat(tune_parts, ignore_index=True) if tune_parts else empty
    out_test_df = pd.concat(test_parts, ignore_index=True) if test_parts else empty
    return tune_df, out_test_df


# ── Vocabulary Building ────────────────────────────────────────────────────────

def build_vocab(
    series: pd.Series,
    max_tokens: int = 10000,
    min_freq: int = 2,
) -> Dict[str, int]:
    """Build a word→index vocabulary from a Series of text.

    Splits on whitespace and non-word characters. Reserves index 0 for <PAD>
    and index 1 for <UNK>.

    Args:
        series: Series of strings to build vocabulary from.
        max_tokens: Maximum vocabulary size (including <PAD> and <UNK>).
        min_freq: Minimum token frequency to include.

    Returns:
        Dict mapping token → integer index.
    """
    counter: Counter = Counter()
    for text in series.dropna():
        tokens = _split_tokens(str(text))
        counter.update(tokens)

    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    # Sort by frequency, then alphabetically for determinism
    sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    for token, count in sorted_tokens:
        if count < min_freq:
            continue
        if len(vocab) >= max_tokens:
            break
        vocab[token] = len(vocab)

    return vocab


def _split_tokens(text: str) -> List[str]:
    """Split a string into tokens on whitespace and non-word boundaries."""
    return re.findall(r"[^\s,;:|!=+]+", text.lower())


# ── Tokenization ───────────────────────────────────────────────────────────────

def tokenize_texts(
    series: pd.Series,
    vocab: Dict[str, int],
    max_len: int = 64,
) -> List[List[int]]:
    """Convert a Series of text into padded token index sequences.

    Args:
        series: Series of strings to tokenize.
        vocab: Token → index mapping from build_vocab.
        max_len: Truncate/pad to this length.

    Returns:
        List of token index lists, each of length max_len.
    """
    unk_idx = vocab.get("<UNK>", 1)
    pad_idx = vocab.get("<PAD>", 0)

    sequences = []
    for text in series.fillna(""):
        tokens = _split_tokens(str(text))
        indices = [vocab.get(t, unk_idx) for t in tokens]
        indices = indices[:max_len]
        indices += [pad_idx] * (max_len - len(indices))
        sequences.append(indices)

    return sequences


# ── Feature Preprocessing ──────────────────────────────────────────────────────

def preprocess_features(
    df: pd.DataFrame,
    process_name_vocab: Optional[Dict[str, int]] = None,
    args_vocab: Optional[Dict[str, int]] = None,
    cat_vocabs: Optional[Dict[str, Dict[int, int]]] = None,
    max_args_len: int = 64,
) -> Dict[str, np.ndarray]:
    """Convert a raw BETH DataFrame into categorical index feature arrays.

    Every feature becomes an int64 class index — embeddings consume them
    directly, and the predicted fields double as cross-entropy targets.
    Bucketing is deterministic (not fitted), and vocabularies are built
    from training data only, so all splits share one schema by construction.

    Features:
      - processName: exact-name class index (0 = OOV)
      - args_ids: token indices (n_events, max_args_len), 0 = <PAD>
      - userId: categorical index (0 = OOV)
      - eventId: categorical index (0 = OOV)
      - argsNum: bucket class in [0, 15]
      - returnValue: bucket class in [0, 12]
      - parentProcessId: 3-level index (0, 1, other)

    Args:
        df: Raw BETH DataFrame.
        process_name_vocab: Exact-name vocab. Built from df if None.
        args_vocab: Token vocab. Built from df if None.
        cat_vocabs: Dict of feature_name → {value: index} for userId and
            eventId. When None, all values map to OOV.
        max_args_len: Max token length for args.

    Returns:
        features_dict of int64 numpy arrays.
    """
    if process_name_vocab is None:
        process_name_vocab = build_process_name_vocab(df["processName"])
    if args_vocab is None:
        args_vocab = build_vocab(df["args"], max_tokens=10000)

    args_ids = tokenize_texts(df["args"], args_vocab, max_len=max_args_len)

    cat_vocabs = cat_vocabs or {}
    user_id = map_categorical(df["userId"], cat_vocabs.get("userId", {})) \
        if "userId" in df.columns else np.zeros(len(df), dtype=np.int64)
    event_id = map_categorical(df["eventId"], cat_vocabs.get("eventId", {})) \
        if "eventId" in df.columns else np.zeros(len(df), dtype=np.int64)

    # parentProcessId: closed 3-level transform — real value 0 (no parent),
    # 1, and everything else. Not an OOV scheme; all three are real classes.
    parent_pid = df["parentProcessId"].fillna(0).astype(np.int64).values
    parent_pid_cat = np.where(parent_pid == 0, 0,
                       np.where(parent_pid == 1, 1, 2)).astype(np.int64)

    features = {
        "processName": map_process_name(df["processName"], process_name_vocab),
        "args_ids": np.array(args_ids, dtype=np.int64),
        "userId": user_id,
        "eventId": event_id,
        "argsNum": bucket_args_num(df["argsNum"]),
        "returnValue": bucket_return_value(df["returnValue"]),
        "parentProcessId": parent_pid_cat,
    }

    return features


# ── Next-Event Targets ─────────────────────────────────────────────────────────

# The five fields predicted by the next-event model, one cross-entropy head
# per field. args content, mountNamespace, and parentProcessId are excluded
# from targets (UNK-flooding / dropped / instance-specific noise) per ADR-0004.
TARGET_FIELDS = ("eventId", "processName", "userId", "returnValue", "argsNum")


# ── Trailing-Window Dataset ────────────────────────────────────────────────────

class TrailingWindowDataset(Dataset):
    """Per-event dataset: trailing 512-event context → next event.

    Event ``i``'s context is the window of ``window_size`` events immediately
    preceding it (positions ``i − window_size … i − 1``); the prediction
    target is event ``i`` itself. Contexts never cross host boundaries, and
    events without a full trailing context within their host are skipped
    (truncate — no padding).

    ``stride`` subsamples target positions (dense stride=1 for evaluation).

    Labels, when provided, are stored (not yielded) so evaluation code can
    look up ``labels[center]`` for the scored events.

    Yields (context, targets) tuples where:
      - context is a dict of int64 tensors, each with ``window_size`` in dim 0
      - targets is a dict of scalar int64 tensors, one per TARGET_FIELDS entry
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        host_lengths: List[int],
        window_size: int = 512,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,
    ):
        self.features = features
        self.host_lengths = host_lengths
        self.window_size = window_size
        self.stride = stride
        self.labels = labels

        # A position is a valid target if it has a full window of preceding
        # events within the same host: first target of a host is at
        # offset + window_size, and the host must have at least one event
        # past it.
        self.centers: List[int] = []
        offset = 0
        for host_n in host_lengths:
            first = offset + window_size
            last = offset + host_n  # exclusive
            self.centers.extend(range(first, last, stride))
            offset += host_n

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        center = self.centers[idx]
        start = center - self.window_size

        context = {
            key: torch.from_numpy(arr[start:center])
            for key, arr in self.features.items()
        }
        targets = {
            field: torch.tensor(int(self.features[field][center]), dtype=torch.int64)
            for field in TARGET_FIELDS
        }
        return context, targets


# ── Utility ────────────────────────────────────────────────────────────────────

def _sort_within_hosts(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Sort each host's events by timestamp, concatenate in deterministic order.

    Keeps each host's events contiguous so trailing windows never span hosts.
    Returns the sorted DataFrame and a list of event counts per host.

    Args:
        df: DataFrame with 'hostName' and optionally 'timestamp' columns.

    Returns:
        (sorted_df, host_lengths) tuple.
    """
    host_lengths = []
    sorted_parts = []
    for host in sorted(df["hostName"].unique()):
        host_df = df[df["hostName"] == host].copy()
        if "timestamp" in host_df.columns:
            host_df.sort_values("timestamp", inplace=True)
        host_lengths.append(len(host_df))
        sorted_parts.append(host_df)
    if sorted_parts:
        return pd.concat(sorted_parts, ignore_index=True), host_lengths
    return df, [len(df)]


# ── End-to-End Pipeline ────────────────────────────────────────────────────────

def load_next_event_pipeline(
    raw_dir: str = "data/raw/per_host",
    window_size: int = 512,
    train_stride: int = 8,
    val_stride: int = 16,
    tune_stride: int = 1,
    n_blocks: int = 50,
    tune_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[
    TrailingWindowDataset,
    TrailingWindowDataset,
    TrailingWindowDataset,
    TrailingWindowDataset,
    Dict[str, int],
    Dict[str, object],
]:
    """End-to-end pipeline for next-event training and honest evaluation.

    The pipeline:
      1. Load per-host CSVs → sort within host → combine
      2. Split by host (attack hosts → test, benign hosts → train/val/test)
      3. Carve the attack-val tuning set out of the test split
      4. Build vocabularies from training data only
      5. Preprocess features (categorical indices, deterministic buckets)
      6. Return TrailingWindowDataset instances + vocabulary artifacts

    Training and validation data are benign hosts only — no attack labels
    anywhere in training (ADR-0003's core principle).

    Args:
        raw_dir: Directory containing per-host BETH CSVs.
        window_size: Events per trailing context (default 512).
        train_stride: Target subsampling for training (default 8).
        val_stride: Target subsampling for benign validation (default 16).
        tune_stride: Target subsampling for the attack-val tuning set
            (default 1 — dense, for threshold candidates).
        n_blocks: Contiguous blocks per test host in the carve-out.
        tune_frac: Probability each carved block lands in the tuning set.
        seed: Random seed for host split and carve-out.

    Returns:
        (train_ds, val_ds, tune_ds, test_ds, vocab_sizes, vocabs)
        Datasets carry ``.labels`` (evil per event) and ``.centers`` (scored
        positions) for evaluation; ``vocabs`` maps
        {'process_name', 'args', 'cat'} → vocabulary dicts.
    """
    # ── Load ────────────────────────────────────────────────────────────────
    host_dfs = load_beth_data(raw_dir)
    if not host_dfs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    # Sort within each host, then concatenate hosts deterministically
    sorted_parts = []
    for host in sorted(host_dfs):
        df = host_dfs[host]
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        sorted_parts.append(df)
    full_df = pd.concat(sorted_parts, ignore_index=True)

    # ── Split by host, then carve attack-val out of test ─────────────────────
    train_df, val_df, test_df = split_by_host(full_df, seed=seed)
    tune_df, test_df = carve_attack_val(
        test_df, n_blocks=n_blocks, tune_frac=tune_frac, seed=seed,
    )

    # ── Sort within hosts (keep hosts contiguous) ────────────────────────────
    train_df, train_host_lengths = _sort_within_hosts(train_df)
    val_df, val_host_lengths = _sort_within_hosts(val_df)
    tune_df, tune_host_lengths = _sort_within_hosts(tune_df)
    test_df, test_host_lengths = _sort_within_hosts(test_df)

    # ── Build vocabs from training data only ─────────────────────────────────
    process_vocab = build_process_name_vocab(train_df["processName"])
    args_vocab = build_vocab(train_df["args"], max_tokens=10000)
    cat_vocabs: Dict[str, Dict[int, int]] = {
        "userId": build_categorical_vocab(train_df["userId"]),
        "eventId": build_categorical_vocab(train_df["eventId"]),
    }

    # +1 for the reserved OOV index on the value→index vocabs; the args
    # token vocab already includes <PAD> and <UNK> in its length.
    vocab_sizes = {
        "process_name_vocab_size": len(process_vocab) + 1,
        "args_vocab_size": len(args_vocab),
        "user_id_vocab_size": len(cat_vocabs["userId"]) + 1,
        "event_id_vocab_size": len(cat_vocabs["eventId"]) + 1,
        "args_num_vocab_size": ARGS_NUM_CLASSES,
        "return_value_vocab_size": RETURN_VALUE_CLASSES,
        "parent_pid_vocab_size": 3,
    }

    vocabs = {
        "process_name": process_vocab,
        "args": args_vocab,
        "cat": cat_vocabs,
    }

    # ── Preprocess ───────────────────────────────────────────────────────────
    train_feat = preprocess_features(train_df, process_vocab, args_vocab, cat_vocabs)
    val_feat = preprocess_features(val_df, process_vocab, args_vocab, cat_vocabs)
    tune_feat = preprocess_features(tune_df, process_vocab, args_vocab, cat_vocabs)
    test_feat = preprocess_features(test_df, process_vocab, args_vocab, cat_vocabs)

    # ── Create datasets ──────────────────────────────────────────────────────
    train_ds = TrailingWindowDataset(
        train_feat, train_host_lengths, window_size, train_stride,
        labels=_get_labels(train_df),
    )
    val_ds = TrailingWindowDataset(
        val_feat, val_host_lengths, window_size, val_stride,
        labels=_get_labels(val_df),
    )
    tune_ds = TrailingWindowDataset(
        tune_feat, tune_host_lengths, window_size, tune_stride,
        labels=_get_labels(tune_df),
    )
    test_ds = TrailingWindowDataset(
        test_feat, test_host_lengths, window_size, 1,  # dense: evaluation
        labels=_get_labels(test_df),
    )

    return train_ds, val_ds, tune_ds, test_ds, vocab_sizes, vocabs
