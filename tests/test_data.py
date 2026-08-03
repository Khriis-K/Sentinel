"""
Tests for src.data — BETH dataset loading, host-based splitting,
text tokenization, and PyTorch Dataset for sliding windows.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    load_beth_data,
    split_by_host,
    build_vocab,
    tokenize_texts,
    preprocess_features,
    BethDataset,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Synthetic BETH-style DataFrame with all 14 fields."""
    n = 2000
    hosts = [f"honeypot-{i}" for i in range(5)]
    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "timestamp": rng.uniform(1.6e9, 1.7e9, n),
        "processId": rng.integers(1, 30000, n),
        "threadId": rng.integers(1, 30000, n),
        "parentProcessId": rng.integers(0, 5, n),
        "userId": rng.integers(0, 10, n),
        "mountNamespace": rng.integers(0, 50, n),
        "processName": rng.choice(["systemd", "sshd", "bash", "curl", "wget", "python3"], n),
        "hostName": rng.choice(hosts, n),
        "eventId": rng.integers(1, 60, n),
        "eventName": rng.choice(["execve", "open", "connect", "write", "read"], n),
        "argsNum": rng.integers(0, 10, n),
        "returnValue": rng.choice([0, 0, 0, -1, 1, 2], n),
        "stackAddresses": rng.choice(["0x7fff", "0x8000", "0x9000", ""], n),
        "args": rng.choice(["-c", "/bin/sh", "-l", "curl http://evil.com", "", "ls -la"], n),
        "sus": rng.integers(0, 2, n),
        "evil": rng.choice([0, 0, 0, 0, 0, 1], n),  # ~16% malicious
    })


@pytest.fixture
def csv_files(sample_df):
    """Write sample_df to temporary CSV files, one per host."""
    tmpdir = tempfile.mkdtemp()
    for host in sample_df["hostName"].unique():
        host_df = sample_df[sample_df["hostName"] == host]
        host_df.to_csv(os.path.join(tmpdir, f"{host}.csv"), index=False)
    yield tmpdir
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def _preprocess(sample_df):
    """Helper: call preprocess_features and return just the feature dict."""
    features, _ = preprocess_features(
        sample_df,
        process_name_vocab=build_vocab(sample_df["processName"], 100),
        args_vocab=build_vocab(sample_df["args"], 200),
    )
    return features


# ── Data Loading ───────────────────────────────────────────────────────────────

def test_load_beth_data_loads_all_csvs(csv_files, sample_df):
    """load_beth_data should load all CSVs from a directory."""
    dfs = load_beth_data(csv_files)
    total = sum(len(df) for df in dfs.values())
    assert total == len(sample_df)


def test_load_beth_data_keys_are_hostnames(csv_files):
    """Returned dict keys should be hostnames (filename without .csv)."""
    dfs = load_beth_data(csv_files)
    for key in dfs:
        assert key.startswith("honeypot-")
        assert not key.endswith(".csv")


def test_load_beth_data_empty_dir(tmp_path):
    """Loading from an empty directory should return empty dict."""
    dfs = load_beth_data(str(tmp_path))
    assert dfs == {}


# ── Host-Based Split ───────────────────────────────────────────────────────────

def test_split_by_host_no_overlap(sample_df):
    """Train, val, and test splits must have no host overlap."""
    train, val, test = split_by_host(
        sample_df, train_frac=0.6, val_frac=0.2, seed=42
    )
    train_hosts = set(train["hostName"].unique())
    val_hosts = set(val["hostName"].unique())
    test_hosts = set(test["hostName"].unique())

    assert train_hosts.isdisjoint(val_hosts)
    assert train_hosts.isdisjoint(test_hosts)
    assert val_hosts.isdisjoint(test_hosts)


def test_split_by_host_preserves_all_rows(sample_df):
    """Splitting should not lose or duplicate any rows."""
    train, val, test = split_by_host(
        sample_df, train_frac=0.6, val_frac=0.2, seed=42
    )
    total_split = len(train) + len(val) + len(test)
    assert total_split == len(sample_df)


def test_split_by_host_test_gets_malicious(sample_df):
    """Test split should contain the malicious host(s). If any host has evil=1,
    it should land in the test set."""
    evil_hosts = set(sample_df[sample_df["evil"] == 1]["hostName"].unique())
    _, _, test = split_by_host(
        sample_df, train_frac=0.6, val_frac=0.2, seed=42
    )
    test_hosts = set(test["hostName"].unique())
    for h in evil_hosts:
        assert h in test_hosts, f"Evil host {h} not in test set"


def test_split_by_host_deterministic(sample_df):
    """Same seed should produce identical splits."""
    t1, v1, te1 = split_by_host(sample_df, seed=123)
    t2, v2, te2 = split_by_host(sample_df, seed=123)
    assert t1.equals(t2)
    assert v1.equals(v2)
    assert te1.equals(te2)


# ── Vocabulary Building ────────────────────────────────────────────────────────

def test_build_vocab_returns_dict():
    """build_vocab returns a token→idx mapping with <PAD> and <UNK>."""
    texts = pd.Series(["curl http://evil.com", "wget -O /tmp/x", "bash -c ls"])
    vocab = build_vocab(texts, max_tokens=100)
    assert isinstance(vocab, dict)
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab


def test_build_vocab_pad_is_zero():
    """<PAD> must be index 0 so padding works correctly."""
    vocab = build_vocab(pd.Series(["a b c"]), max_tokens=50)
    assert vocab["<PAD>"] == 0


def test_build_vocab_unk_is_one():
    """<UNK> must be index 1."""
    vocab = build_vocab(pd.Series(["a b c"]), max_tokens=50)
    assert vocab["<UNK>"] == 1


def test_build_vocab_respects_max_tokens():
    """Vocabulary size should not exceed max_tokens."""
    texts = pd.Series([" ".join(str(i) for i in range(1000))])
    vocab = build_vocab(texts, max_tokens=50)
    assert len(vocab) <= 50


# ── Tokenization ───────────────────────────────────────────────────────────────

def test_tokenize_texts_output_shape():
    """Output should be a list of lists, each of length max_len."""
    texts = pd.Series(["curl http://evil.com", "bash"])
    vocab = build_vocab(texts, max_tokens=100)
    tokens = tokenize_texts(texts, vocab, max_len=10)
    assert len(tokens) == len(texts)
    for seq in tokens:
        assert len(seq) == 10


def test_tokenize_texts_unknown_words_map_to_unk():
    """Words not in vocab should map to <UNK> index."""
    texts = pd.Series(["knownword"])
    vocab = build_vocab(texts, max_tokens=10)
    tokens = tokenize_texts(pd.Series(["knownword totallynew"]), vocab, max_len=5)
    unk_idx = vocab["<UNK>"]
    assert unk_idx in tokens[0]


# ── Feature Preprocessing ──────────────────────────────────────────────────────

def test_preprocess_features_output_keys(sample_df):
    """preprocess_features should return a feature dict and numeric stats."""
    features, stats = preprocess_features(
        sample_df,
        process_name_vocab=build_vocab(sample_df["processName"], 100),
        args_vocab=build_vocab(sample_df["args"], 200),
        max_args_len=20,
    )
    expected_keys = {
        "processName_ids", "args_ids",
        "userId", "mountNamespace", "eventId",
        "argsNum", "returnValue", "parentProcessId",
    }
    assert set(features.keys()) == expected_keys
    # stats should cover the numeric features
    assert set(stats.keys()) == {"argsNum", "returnValue", "parentProcessId"}


def test_preprocess_features_numeric_normalized(sample_df):
    """Numeric features should be zero-mean, unit-variance (roughly)."""
    features, _ = preprocess_features(
        sample_df,
        process_name_vocab=build_vocab(sample_df["processName"], 100),
        args_vocab=build_vocab(sample_df["args"], 200),
    )
    assert abs(np.mean(features["argsNum"])) < 1e-5
    assert abs(np.std(features["argsNum"]) - 1.0) < 0.01


def test_preprocess_features_reuses_stats(sample_df):
    """When numeric_stats is provided, val/test should use train's stats."""
    train_df = sample_df.iloc[:1000]
    val_df = sample_df.iloc[1000:]

    _, train_stats = preprocess_features(
        train_df,
        process_name_vocab=build_vocab(train_df["processName"], 100),
        args_vocab=build_vocab(train_df["args"], 200),
    )
    val_feat, val_stats = preprocess_features(
        val_df,
        process_name_vocab=build_vocab(train_df["processName"], 100),
        args_vocab=build_vocab(train_df["args"], 200),
        numeric_stats=train_stats,
    )
    # val_stats should equal train_stats (not recomputed from val)
    for key in train_stats:
        assert val_stats[key] == train_stats[key], f"{key} stats differ"


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

def test_beth_dataset_yields_windows(sample_df):
    """BethDataset should yield (features, label) tuples with correct shapes."""
    features = _preprocess(sample_df)
    ds = BethDataset(features, sample_df["evil"].values, window_size=128, stride=64)
    x, y = ds[0]

    assert isinstance(x, dict)
    assert isinstance(y, (int, float, np.integer, np.floating, torch.Tensor))
    for key, arr in x.items():
        assert arr.shape[0] == 128, f"{key} shape {arr.shape} — expected 128 in dim 0"


def test_beth_dataset_label_is_binary(sample_df):
    """Labels should be 0 (benign) or 1 (malicious)."""
    features = _preprocess(sample_df)
    ds = BethDataset(features, sample_df["evil"].values, window_size=128, stride=64)
    for i in range(min(20, len(ds))):
        _, y = ds[i]
        assert y in (0, 1), f"Label {y} at index {i} is not binary"


def test_beth_dataset_len(sample_df):
    """Dataset length should match the number of sliding windows."""
    features = _preprocess(sample_df)
    window_size = 128
    stride = 64

    ds = BethDataset(features, sample_df["evil"].values, window_size=window_size, stride=stride)
    n_events = len(sample_df)
    expected_windows = max(0, (n_events - window_size) // stride + 1)
    assert len(ds) == expected_windows


def test_beth_dataset_short_sequence():
    """Dataset with fewer events than window_size should be empty."""
    df = pd.DataFrame({
        "timestamp": range(50),
        "processId": range(50),
        "threadId": range(50),
        "parentProcessId": [0] * 50,
        "userId": [1] * 50,
        "mountNamespace": [1] * 50,
        "processName": ["bash"] * 50,
        "hostName": ["test-host"] * 50,
        "eventId": [1] * 50,
        "eventName": ["execve"] * 50,
        "argsNum": [2] * 50,
        "returnValue": [0] * 50,
        "stackAddresses": [""] * 50,
        "args": ["-c"] * 50,
        "sus": [0] * 50,
        "evil": [0] * 50,
    })
    features, _ = preprocess_features(
        df,
        process_name_vocab=build_vocab(df["processName"], 10),
        args_vocab=build_vocab(df["args"], 10),
    )
    ds = BethDataset(features, df["evil"].values, window_size=512, stride=256)
    assert len(ds) == 0
