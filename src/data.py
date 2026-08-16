"""
Sentinel — Data Pipeline
BETH dataset loading, host-based splitting, text tokenization,
feature preprocessing, and PyTorch Dataset for sliding windows.
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

NUMERIC_FEATURES = ["argsNum", "returnValue", "parentProcessId"]
METRIC_KEYS = ["auroc", "pr_auc", "f1", "precision", "recall"]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_beth_data(raw_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from a directory, keyed by hostname.

    Each CSV is named <hostname>.csv as in the BETH dataset convention.
    Returns a dict mapping hostname → DataFrame.

    Args:
        raw_dir: Path to directory containing BETH CSV files.

    Returns:
        Dict of hostname → DataFrame for each CSV found.
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

    Maps unique integer values to contiguous indices, reserving 0 for unseen/OOV.
    Useful for high-cardinality integer features like mountNamespace or userId
    where raw values can be sparse 32-bit integers.

    Args:
        series: Integer feature values.
        max_size: Maximum number of unique values to keep (most frequent).

    Returns:
        Dict mapping original integer value → contiguous index.
    """
    value_counts = series.fillna(0).astype(np.int64).value_counts()
    vocab: Dict[int, int] = {0: 0}  # reserve 0 for unseen

    for val in value_counts.index:
        if len(vocab) >= max_size:
            break
        int_val = int(val)
        if int_val == 0:
            continue  # already mapped
        vocab[int_val] = len(vocab)

    return vocab


def map_categorical(series: pd.Series, vocab: Dict[int, int]) -> np.ndarray:
    """Map a Series of integer values through a categorical vocabulary.

    Values not in the vocab default to index 0.

    Args:
        series: Integer feature values.
        vocab: Value → index mapping from build_categorical_vocab.

    Returns:
        int64 numpy array of mapped indices.
    """
    default = vocab.get(0, 0)
    return np.array(
        [vocab.get(int(v), default) for v in series.fillna(0)],
        dtype=np.int64,
    )


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
    max_args_len: int = 64,
    max_process_name_len: int = 16,
    numeric_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    cat_vocabs: Optional[Dict[str, Dict[int, int]]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Convert a raw BETH DataFrame into numeric feature arrays.

    Returns feature dict and numeric_stats dict (mean, std per numeric feature).
    If numeric_stats is provided, use those for standardization instead of
    computing from df — this ensures val/test use train's statistics.

    Features:
      - processName_ids: token indices (n_events, max_process_name_len)
      - args_ids: token indices (n_events, max_args_len)
      - userId: integer array
      - mountNamespace: integer array
      - eventId: integer array
      - argsNum: float32 (standardized)
      - returnValue: float32 (standardized)
      - parentProcessId: float32 (3-level → standardized)

    Args:
        df: Raw BETH DataFrame.
        process_name_vocab: Vocab for processName. Built from data if None.
        args_vocab: Vocab for args. Built from data if None.
        max_args_len: Max token length for args.
        max_process_name_len: Max token length for processName.
        numeric_stats: Optional dict of feature_name → (mean, std) for
            standardization. If None, stats are computed from df.
        cat_vocabs: Optional dict of feature_name → {value: index} for
            categorical features (userId, mountNamespace, eventId).
            When provided, raw values are mapped to contiguous indices.
            When None, raw integer values are used as-is.

    Returns:
        (features_dict, numeric_stats_dict)
    """
    # Build vocabs if not provided
    if process_name_vocab is None:
        process_name_vocab = build_vocab(df["processName"], max_tokens=5000)
    if args_vocab is None:
        args_vocab = build_vocab(df["args"], max_tokens=10000)

    # Tokenize text features
    process_name_ids = tokenize_texts(
        df["processName"], process_name_vocab, max_len=max_process_name_len
    )
    args_ids = tokenize_texts(
        df["args"], args_vocab, max_len=max_args_len
    )

    # Categorical features — map to contiguous indices if vocabs provided
    cat_vocabs = cat_vocabs or {}
    user_id = map_categorical(df["userId"], cat_vocabs.get("userId", {})) \
        if "userId" in cat_vocabs and "userId" in df.columns \
        else np.zeros(len(df), dtype=np.int64)
    mount_ns = map_categorical(df["mountNamespace"], cat_vocabs.get("mountNamespace", {})) \
        if "mountNamespace" in cat_vocabs and "mountNamespace" in df.columns \
        else np.zeros(len(df), dtype=np.int64)
    event_id = map_categorical(df["eventId"], cat_vocabs.get("eventId", {})) \
        if "eventId" in cat_vocabs and "eventId" in df.columns \
        else np.zeros(len(df), dtype=np.int64)

    # Numeric features — 3-level parentProcessId + argsNum + returnValue
    parent_pid = df["parentProcessId"].fillna(0).astype(np.float64).values
    parent_pid_cat = np.where(parent_pid == 0, 0.0,
                      np.where(parent_pid == 1, 1.0, 2.0))

    raw_numeric = {
        "argsNum": df["argsNum"].fillna(0).astype(np.float64).values,
        "returnValue": df["returnValue"].fillna(0).astype(np.float64).values,
        "parentProcessId": parent_pid_cat,
    }

    # Standardize: use provided stats or compute from this df
    computed_stats = {}
    standardized = {}
    for name in NUMERIC_FEATURES:
        if numeric_stats is not None and name in numeric_stats:
            mean, std = numeric_stats[name]
        else:
            mean = float(raw_numeric[name].mean())
            std = float(raw_numeric[name].std())
        computed_stats[name] = (mean, std)
        standardized[name] = ((raw_numeric[name] - mean) / (std + 1e-8)).astype(np.float32)

    features = {
        "processName_ids": np.array(process_name_ids, dtype=np.int64),
        "args_ids": np.array(args_ids, dtype=np.int64),
        "userId": user_id,
        "mountNamespace": mount_ns,
        "eventId": event_id,
        "argsNum": standardized["argsNum"],
        "returnValue": standardized["returnValue"],
        "parentProcessId": standardized["parentProcessId"],
    }

    return features, computed_stats


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class BethDataset(Dataset):
    """Sliding-window dataset over BETH feature arrays.

    Yields (features, label) tuples where:
      - features is a dict of torch tensors, each of shape (window_size, ...)
      - label is 0 (benign) or 1 (malicious)

    A window is labeled malicious if it contains at least one evil==1 event.
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
        window_size: int = 512,
        stride: int = 256,
    ):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.stride = stride

        n_events = len(labels)
        if n_events < window_size:
            self.n_windows = 0
        else:
            self.n_windows = (n_events - window_size) // stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size

        x = {}
        for key, arr in self.features.items():
            window = arr[start:end].copy()
            x[key] = torch.from_numpy(window)

        window_labels = self.labels[start:end]
        label = 1 if np.any(window_labels == 1) else 0

        return x, torch.tensor(label, dtype=torch.float32)


# ── Centered-Window Dataset ────────────────────────────────────────────────────

class CenteredWindowDataset(Dataset):
    """Per-event dataset using centered windows for training and evaluation.

    Each window is a 512-event slice centered on a specific event at position i
    (256 before, 255 after, plus the event itself). The label is ``evil[i]`` —
    the center event's value — not ``any(evil_in_window)``.

    Host boundaries are respected: windows never span across different hosts.
    Edge handling is truncate — events without 256 neighbors on both sides
    within their host are not used as centers.

    Yields (features, label) tuples where:
      - features is a dict of torch tensors, each of shape (window_size, ...)
      - label is 0 (benign center) or 1 (malicious center)
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
        host_lengths: List[int],
        window_size: int = 512,
        stride: int = 32,
    ):
        self.features = features
        self.labels = labels
        self.host_lengths = host_lengths
        self.window_size = window_size
        self.stride = stride
        self.half = window_size // 2

        # Build the list of global indices that are valid center positions.
        # A position is valid if it has `half` events before and after it
        # within the same host.
        self.centers: List[int] = []
        offset = 0
        for host_n in host_lengths:
            if host_n >= window_size:
                first_center = offset + self.half
                last_center = offset + host_n - self.half
                # range stop is exclusive, so we go up to last_center inclusive
                for center in range(first_center, last_center, stride):
                    self.centers.append(center)
            offset += host_n

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        center = self.centers[idx]
        start = center - self.half
        end = center + self.half

        x = {}
        for key, arr in self.features.items():
            window = arr[start:end].copy()
            x[key] = torch.from_numpy(window)

        label = self.labels[center]
        return x, torch.tensor(label, dtype=torch.float32)


# ── Utility ────────────────────────────────────────────────────────────────────

def load_and_prepare(
    raw_dir: str,
    window_size: int = 512,
    stride: int = 256,
    seed: int = 42,
) -> Tuple[BethDataset, BethDataset, BethDataset, Dict, Dict]:
    """End-to-end pipeline: load BETH data, split by host, preprocess, create datasets.

    Events are sorted by timestamp within each host before windowing,
    ensuring windows represent contiguous event sequences.

    Returns:
        (train_ds, val_ds, test_ds, process_name_vocab, args_vocab)
    """
    host_dfs = load_beth_data(raw_dir)
    if not host_dfs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    # Sort each host's events by timestamp, then combine
    sorted_dfs = []
    for host, df in host_dfs.items():
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        sorted_dfs.append(df)

    full_df = pd.concat(sorted_dfs, ignore_index=True)

    # Split by host
    train_df, val_df, test_df = split_by_host(full_df, seed=seed)

    # Build vocabs from training data only
    process_vocab = build_vocab(train_df["processName"], max_tokens=5000)
    args_vocab = build_vocab(train_df["args"], max_tokens=10000)

    # Preprocess — fit on train, transform val/test with train stats
    train_feat, numeric_stats = preprocess_features(
        train_df, process_vocab, args_vocab,
    )
    val_feat, _ = preprocess_features(
        val_df, process_vocab, args_vocab, numeric_stats=numeric_stats,
    )
    test_feat, _ = preprocess_features(
        test_df, process_vocab, args_vocab, numeric_stats=numeric_stats,
    )

    # Create datasets
    train_ds = BethDataset(
        train_feat, _get_labels(train_df),
        window_size=window_size, stride=stride,
    )
    val_ds = BethDataset(
        val_feat, _get_labels(val_df),
        window_size=window_size, stride=stride,
    )
    test_ds = BethDataset(
        test_feat, _get_labels(test_df),
        window_size=window_size, stride=stride,
    )

    return train_ds, val_ds, test_ds, process_vocab, args_vocab


# ── Benchmark Split Loader ────────────────────────────────────────────────────

def load_benchmark_splits(
    data_dir: str,
    window_size: int = 512,
    stride: int = 256,
    train_attack_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[BethDataset, BethDataset, BethDataset, Dict, Dict, Dict[str, int]]:
    """Load the 3 pre-split BETH benchmark CSV files and create PyTorch Datasets.

    The BETH benchmark splits are designed for unsupervised anomaly detection:
    training contains only benign hosts, the test host contains the attack.
    For supervised BiLSTM training, a fraction of the test host's events are
    mixed into the training set so the model sees positive examples.

    Args:
        data_dir: Directory containing labelled_{training,validation,testing}_data.csv.
        window_size: Events per sliding window.
        stride: Stride between windows.
        train_attack_frac: Fraction of the test host's events to mix into training
            for supervised learning (default 0.2 = 20%).
        seed: Random seed for attack-data split.

    Returns:
        (train_ds, val_ds, test_ds, process_vocab, args_vocab, vocab_sizes)
        where vocab_sizes maps embedding key → vocabulary size.
    """
    import os

    train_path = os.path.join(data_dir, "labelled_training_data.csv")
    val_path = os.path.join(data_dir, "labelled_validation_data.csv")
    test_path = os.path.join(data_dir, "labelled_testing_data.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Benchmark file not found: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Sort by timestamp within each split
    for df in [train_df, val_df, test_df]:
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True)

    # Mix attack data into training for supervised learning.
    # The test host contains the only evil==1 events.
    if train_attack_frac > 0 and "evil" in test_df.columns:
        rng = np.random.default_rng(seed)
        test_hosts = test_df["hostName"].unique()
        n_test = len(test_df)
        n_mix = int(n_test * train_attack_frac)

        # Shuffle indices and split
        indices = rng.permutation(n_test)
        mix_idx = indices[:n_mix]
        keep_idx = indices[n_mix:]

        mix_df = test_df.iloc[mix_idx].copy()
        test_df = test_df.iloc[keep_idx].copy()

        train_df = pd.concat([train_df, mix_df], ignore_index=True)
        # Re-sort training by timestamp
        train_df.sort_values("timestamp", inplace=True)

    # Build vocabs from training data only
    process_vocab = build_vocab(train_df["processName"], max_tokens=5000)
    args_vocab = build_vocab(train_df["args"], max_tokens=10000)

    # Build categorical vocabs for high-cardinality integer features
    cat_vocabs = {
        "userId": build_categorical_vocab(train_df["userId"]),
        "mountNamespace": build_categorical_vocab(train_df["mountNamespace"]),
        "eventId": build_categorical_vocab(train_df["eventId"]),
    }

    # Vocabulary sizes for categorical embedding layers
    vocab_sizes = {
        "processName": len(process_vocab),
        "args": len(args_vocab),
        "userId": len(cat_vocabs["userId"]),
        "mountNamespace": len(cat_vocabs["mountNamespace"]),
        "eventId": len(cat_vocabs["eventId"]),
    }

    # Preprocess — fit on train, transform val/test with train stats
    train_feat, numeric_stats = preprocess_features(
        train_df, process_vocab, args_vocab, cat_vocabs=cat_vocabs,
    )
    val_feat, _ = preprocess_features(
        val_df, process_vocab, args_vocab, numeric_stats=numeric_stats, cat_vocabs=cat_vocabs,
    )
    test_feat, _ = preprocess_features(
        test_df, process_vocab, args_vocab, numeric_stats=numeric_stats, cat_vocabs=cat_vocabs,
    )

    # Create datasets
    train_ds = BethDataset(
        train_feat, _get_labels(train_df),
        window_size=window_size, stride=stride,
    )
    val_ds = BethDataset(
        val_feat, _get_labels(val_df),
        window_size=window_size, stride=stride,
    )
    test_ds = BethDataset(
        test_feat, _get_labels(test_df),
        window_size=window_size, stride=stride,
    )

    return train_ds, val_ds, test_ds, process_vocab, args_vocab, vocab_sizes


# ── Per-Host Pipeline ──────────────────────────────────────────────────────────

def _sort_within_hosts(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Sort each host's events by timestamp, concatenate in deterministic order.

    Keeps each host's events contiguous so centered windows never span hosts.
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


def load_per_host_pipeline(
    raw_dir: str = "data/raw/per_host",
    window_size: int = 512,
    stride: int = 32,
    train_attack_frac: float = 0.2,
    val_attack_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[
    "CenteredWindowDataset",
    "CenteredWindowDataset",
    "CenteredWindowDataset",
    Optional["CenteredWindowDataset"],
    Dict[str, int],
    Dict[str, int],
    Dict[str, Dict[int, int]],
    Dict[str, Tuple[float, float]],
    Dict[str, int],
]:
    """End-to-end pipeline using per-host CSVs with centered-window datasets.

    Loads CSVs from ``raw_dir``, splits by host, mixes attack data for supervised
    training, and returns ``CenteredWindowDataset`` instances with per-event labels.

    The pipeline:
      1. Load per-host CSVs → sort within host → combine
      2. Split by host (attack hosts → test, benign hosts → train/val)
      3. Mix ``train_attack_frac`` of test data into training; hold out
         ``val_attack_frac`` of the mixed portion as a mixed-class val set
      4. Build vocabs + categorical vocabs from training data
      5. Preprocess features; track host boundaries
      6. Return ``CenteredWindowDataset`` instances + vocabulary artifacts

    Args:
        raw_dir: Directory containing per-host BETH CSVs (``<hostname>.csv``).
        window_size: Events per centered window (default 512).
        stride: Stride between training centers (default 32).
        train_attack_frac: Fraction of test-host events to mix into training
            for supervised learning (default 0.2).
        val_attack_frac: Fraction of the mixed-in attack data to hold out as
            a mixed-class validation set (default 0.2 → 4 % of original test).
        seed: Random seed for host split and attack-data splits.

    Returns:
        (train_ds, val_ds, test_ds, mixed_val_ds, process_vocab, args_vocab,
         cat_vocabs, numeric_stats, vocab_sizes)
        ``mixed_val_ds`` is ``None`` when ``train_attack_frac == 0`` or no evil
        events exist in the test split.
    """
    # ── Load ────────────────────────────────────────────────────────────────
    host_dfs = load_beth_data(raw_dir)
    if not host_dfs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    # Sort within each host, then concatenate hosts deterministically
    sorted_parts = []
    all_host_lengths = []
    for host in sorted(host_dfs):
        df = host_dfs[host]
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        all_host_lengths.append(len(df))
        sorted_parts.append(df)
    full_df = pd.concat(sorted_parts, ignore_index=True)

    # ── Split by host ───────────────────────────────────────────────────────
    train_df, val_df, test_df = split_by_host(full_df, seed=seed)

    mixed_val_df = None

    # ── Mix attack data for supervised training ──────────────────────────────
    if train_attack_frac > 0 and "evil" in test_df.columns:
        rng = np.random.default_rng(seed)
        n_test = len(test_df)
        n_mix = int(n_test * train_attack_frac)

        indices = rng.permutation(n_test)
        mix_idx = indices[:n_mix]
        keep_idx = indices[n_mix:]

        mix_pool = test_df.iloc[mix_idx].copy()
        test_df = test_df.iloc[keep_idx].copy()

        # Split mix_pool: (1 - val_attack_frac) → training, val_attack_frac → mixed val
        if val_attack_frac > 0:
            rng2 = np.random.default_rng(seed + 1)
            n_pool = len(mix_pool)
            n_train_mix = int(n_pool * (1.0 - val_attack_frac))
            pool_indices = rng2.permutation(n_pool)
            train_mix_idx = pool_indices[:n_train_mix]
            mixed_val_idx = pool_indices[n_train_mix:]

            train_mix = mix_pool.iloc[train_mix_idx].copy()
            mixed_val_df = mix_pool.iloc[mixed_val_idx].copy()
            train_df = pd.concat([train_df, train_mix], ignore_index=True)
        else:
            train_df = pd.concat([train_df, mix_pool], ignore_index=True)

    # ── Sort within hosts (keep hosts contiguous) ────────────────────────────
    train_df, train_host_lengths = _sort_within_hosts(train_df)
    val_df, val_host_lengths = _sort_within_hosts(val_df)
    test_df, test_host_lengths = _sort_within_hosts(test_df)

    if mixed_val_df is not None:
        mixed_val_df, mixed_val_host_lengths = _sort_within_hosts(mixed_val_df)

    # ── Build vocabs from training data only ─────────────────────────────────
    process_vocab = build_vocab(train_df["processName"], max_tokens=5000)
    args_vocab = build_vocab(train_df["args"], max_tokens=10000)

    cat_vocabs: Dict[str, Dict[int, int]] = {}
    for col in ["userId", "mountNamespace", "eventId"]:
        if col in train_df.columns:
            cat_vocabs[col] = build_categorical_vocab(train_df[col])
        else:
            cat_vocabs[col] = {0: 0}  # default single-entry vocab for missing column

    vocab_sizes = {
        "processName": len(process_vocab),
        "args": len(args_vocab),
        "userId": len(cat_vocabs["userId"]),
        "mountNamespace": len(cat_vocabs["mountNamespace"]),
        "eventId": len(cat_vocabs["eventId"]),
    }

    # ── Preprocess ───────────────────────────────────────────────────────────
    train_feat, numeric_stats = preprocess_features(
        train_df, process_vocab, args_vocab, cat_vocabs=cat_vocabs,
    )
    val_feat, _ = preprocess_features(
        val_df, process_vocab, args_vocab, numeric_stats=numeric_stats, cat_vocabs=cat_vocabs,
    )
    test_feat, _ = preprocess_features(
        test_df, process_vocab, args_vocab, numeric_stats=numeric_stats, cat_vocabs=cat_vocabs,
    )

    mixed_val_feat = None
    if mixed_val_df is not None:
        mixed_val_feat, _ = preprocess_features(
            mixed_val_df, process_vocab, args_vocab,
            numeric_stats=numeric_stats, cat_vocabs=cat_vocabs,
        )

    # ── Create datasets ──────────────────────────────────────────────────────
    train_ds = CenteredWindowDataset(
        train_feat, _get_labels(train_df), train_host_lengths,
        window_size=window_size, stride=stride,
    )
    val_ds = CenteredWindowDataset(
        val_feat, _get_labels(val_df), val_host_lengths,
        window_size=window_size, stride=stride,
    )
    test_ds = CenteredWindowDataset(
        test_feat, _get_labels(test_df), test_host_lengths,
        window_size=window_size, stride=stride,
    )

    mixed_val_ds = None
    if mixed_val_feat is not None and mixed_val_df is not None:
        mixed_val_ds = CenteredWindowDataset(
            mixed_val_feat, _get_labels(mixed_val_df), mixed_val_host_lengths,
            window_size=window_size, stride=stride,
        )

    return (
        train_ds, val_ds, test_ds, mixed_val_ds,
        process_vocab, args_vocab, cat_vocabs, numeric_stats, vocab_sizes,
    )
