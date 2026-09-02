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
    build_categorical_vocab,
    map_categorical,
    bucket_return_value,
    bucket_args_num,
    TrailingWindowDataset,
    TARGET_FIELDS,
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


# ── Categorical Vocabulary ─────────────────────────────────────────────────────

def test_categorical_vocab_real_zero_gets_own_index():
    """Real value 0 must map to its own index, distinct from the OOV index.

    Root (userId=0) is 95-99% of BETH rows — if 0 collided with OOV,
    root would be indistinguishable from unseen users.
    """
    series = pd.Series([0, 0, 0, 1, 1, 2])
    vocab = build_categorical_vocab(series)

    assert 0 in vocab, "Real value 0 must be present in the vocab"
    assert vocab[0] != 0, "Real value 0 must not collide with the OOV index"


def test_categorical_vocab_unseen_maps_to_oov():
    """Values not in the vocab must map to the reserved OOV index 0."""
    series = pd.Series([0, 0, 5, 5, 7])
    vocab = build_categorical_vocab(series)

    mapped = map_categorical(pd.Series([0, 5, 7, 999]), vocab)
    assert mapped[0] != 0, "Seen value 0 must not map to OOV"
    assert mapped[1] != 0 and mapped[2] != 0, "Seen values must not map to OOV"
    assert mapped[3] == 0, "Unseen value 999 must map to OOV index 0"


def test_categorical_vocab_deterministic():
    """Same input series must produce an identical vocab, ties included."""
    series = pd.Series([3, 3, 3, 1, 1, 2, 2, 9, 9, 9])
    v1 = build_categorical_vocab(series)
    v2 = build_categorical_vocab(series)
    assert v1 == v2


def test_categorical_vocab_indices_are_contiguous():
    """Assigned indices must be 1..n with no gaps (0 reserved for OOV)."""
    series = pd.Series([10, 10, 20, 20, 30, 40])
    vocab = build_categorical_vocab(series)
    indices = sorted(vocab.values())
    assert indices == list(range(1, len(vocab) + 1))


# ── Numeric Bucketizers ────────────────────────────────────────────────────────

def test_bucket_return_value_error_is_class_zero():
    """returnValue −1 (error) must map to class 0."""
    out = bucket_return_value(pd.Series([-1, -1, -5]))
    np.testing.assert_array_equal(out, [0, 0, 0])


def test_bucket_return_value_zero_is_class_one():
    """returnValue 0 (success) must map to class 1."""
    out = bucket_return_value(pd.Series([0, 0, 0]))
    np.testing.assert_array_equal(out, [1, 1, 1])


def test_bucket_return_value_one():
    """returnValue 1 must land in the first positive bin."""
    out = bucket_return_value(pd.Series([1]))
    assert out[0] >= 2
    assert out[0] <= 12


def test_bucket_return_value_extreme_tail_capped():
    """Huge positive values must saturate at the top class (12)."""
    out = bucket_return_value(pd.Series([10**9, 2**31 - 1]))
    np.testing.assert_array_equal(out, [12, 12])


def test_bucket_return_value_monotonic_in_value():
    """Larger positive return values must never map to a lower class."""
    values = pd.Series([1, 2, 7, 100, 10_000, 10**9])
    out = bucket_return_value(values)
    assert (np.diff(out) >= 0).all(), f"Non-monotonic buckets: {out}"


def test_bucket_return_value_classes_within_13():
    """All outputs must be within [0, 12]."""
    rng = np.random.default_rng(0)
    values = pd.Series(rng.integers(-5, 2**31 - 1, 5000))
    out = bucket_return_value(values)
    assert out.min() >= 0 and out.max() <= 12


def test_bucket_return_value_deterministic():
    """Same input must produce identical buckets."""
    values = pd.Series([-1, 0, 1, 5, 999999])
    out1 = bucket_return_value(values)
    out2 = bucket_return_value(values)
    np.testing.assert_array_equal(out1, out2)


def test_bucket_return_value_handles_nan():
    """NaN must map to the zero/success class (fillna(0) convention)."""
    out = bucket_return_value(pd.Series([np.nan, 0]))
    np.testing.assert_array_equal(out, [1, 1])


def test_bucket_args_num_basic():
    """argsNum maps to its own class for 0..15."""
    values = pd.Series(range(16))
    out = bucket_args_num(values)
    np.testing.assert_array_equal(out, np.arange(16))


def test_bucket_args_num_capped_at_fifteen():
    """Counts of 15 and above must collapse into the top class (15)."""
    out = bucket_args_num(pd.Series([14, 15, 16, 100, 10**9]))
    np.testing.assert_array_equal(out, [14, 15, 15, 15, 15])


def test_bucket_args_num_negative_clamped():
    """Negative counts (shouldn't occur, but) must clamp to class 0."""
    out = bucket_args_num(pd.Series([-3, -1, 0]))
    np.testing.assert_array_equal(out, [0, 0, 0])


def test_bucket_args_num_deterministic():
    """Same input must produce identical buckets."""
    values = pd.Series([0, 3, 15, 42])
    out1 = bucket_args_num(values)
    out2 = bucket_args_num(values)
    np.testing.assert_array_equal(out1, out2)


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


# ═══════════════════════════════════════════════════════════════════════════════
# TrailingWindowDataset Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_features(n_events: int):
    """Build a synthetic feature dict matching the next-event preprocessing."""
    return {
        "processName": np.zeros(n_events, dtype=np.int64),
        "args_ids": np.zeros((n_events, 64), dtype=np.int64),
        "userId": np.zeros(n_events, dtype=np.int64),
        "eventId": np.zeros(n_events, dtype=np.int64),
        "argsNum": np.zeros(n_events, dtype=np.int64),
        "returnValue": np.zeros(n_events, dtype=np.int64),
        "parentProcessId": np.zeros(n_events, dtype=np.int64),
    }


def test_trailing_dataset_single_host():
    """Single host: every event with a full trailing context is a target."""
    n = 1024
    features = _make_features(n)
    host_lengths = [n]
    ds = TrailingWindowDataset(features, host_lengths, window_size=512, stride=1)
    # Valid targets: positions 512 .. 1023 (512 positions)
    assert len(ds) == 512


def test_trailing_dataset_stride_one():
    """Stride=1 should produce one example per valid target."""
    n = 1024
    features = _make_features(n)
    ds = TrailingWindowDataset(features, [n], window_size=512, stride=1)
    assert len(ds) == 512


def test_trailing_dataset_host_too_small():
    """A host with no event past the first full window contributes nothing."""
    features = _make_features(300)
    ds = TrailingWindowDataset(features, [300], window_size=512, stride=1)
    assert len(ds) == 0


def test_trailing_dataset_exactly_window_size():
    """A host with exactly window_size events has no target (needs one more)."""
    features = _make_features(512)
    ds = TrailingWindowDataset(features, [512], window_size=512, stride=1)
    assert len(ds) == 0


def test_trailing_dataset_multiple_hosts():
    """Windows must not cross host boundaries."""
    n_per_host = 600
    features = _make_features(n_per_host * 2)
    ds = TrailingWindowDataset(
        features, [n_per_host, n_per_host], window_size=512, stride=32
    )
    # Each host: targets 512..599 (88 positions), stride 32 → 3 each
    assert len(ds) == 6


def test_trailing_dataset_mixed_host_sizes():
    """Only hosts with room for context + target contribute examples."""
    features = _make_features(1100)  # 600 + 500
    ds = TrailingWindowDataset(features, [600, 500], window_size=512, stride=32)
    # Host 1 (600): 3 windows; Host 2 (500): too small
    assert len(ds) == 3


def test_trailing_dataset_first_center_is_window_size():
    """First target of a host must sit exactly one window into the stream."""
    n = 700
    features = _make_features(n)
    ds = TrailingWindowDataset(features, [n], window_size=512, stride=1)
    assert ds.centers[0] == 512
    assert ds.centers[-1] == n - 1


def test_trailing_dataset_window_content_matches_slice():
    """Context tensors must equal the trailing slice before the target."""
    n = 1024
    features = _make_features(n)
    features["argsNum"] = np.arange(n, dtype=np.int64)
    ds = TrailingWindowDataset(features, [n], window_size=512, stride=1)

    center = 800
    idx = ds.centers.index(center)
    context, targets = ds[idx]
    expected = np.arange(center - 512, center, dtype=np.int64)
    np.testing.assert_array_equal(context["argsNum"].numpy(), expected)
    assert targets["argsNum"].item() == center


def test_trailing_dataset_targets_are_center_event_fields():
    """Target dict must carry the five predicted fields at the target position."""
    n = 1024
    features = _make_features(n)
    features["eventId"] = np.arange(n, dtype=np.int64)
    features["processName"] = np.arange(n, dtype=np.int64) % 50
    features["userId"] = np.arange(n, dtype=np.int64) % 10
    features["returnValue"] = np.full(n, 3, dtype=np.int64)
    features["argsNum"] = np.arange(n, dtype=np.int64) % 16
    ds = TrailingWindowDataset(features, [n], window_size=512, stride=1)

    center = 900
    idx = ds.centers.index(center)
    _, targets = ds[idx]
    assert set(targets.keys()) == set(TARGET_FIELDS)
    for field in TARGET_FIELDS:
        assert targets[field].item() == int(features[field][center])


def test_trailing_dataset_window_never_crosses_hosts():
    """No context may reach into the neighbouring host's events."""
    n1, n2 = 600, 600
    features = _make_features(n1 + n2)
    features["argsNum"] = np.arange(n1 + n2, dtype=np.int64)
    ds = TrailingWindowDataset(features, [n1, n2], window_size=512, stride=1)

    for i in range(len(ds)):
        center = ds.centers[i]
        start = center - 512
        if center < n1:
            assert start >= 0, f"Host-1 window starts at {start} (before stream)"
        else:
            assert start >= n1, (
                f"Window for target {center} starts at {start}, "
                f"crossing into host 1 (boundary at {n1})"
            )


def test_trailing_dataset_context_length_is_window_size():
    """Every context must have exactly window_size events in dim 0."""
    n = 1024
    features = _make_features(n)
    ds = TrailingWindowDataset(features, [n], window_size=512, stride=7)
    for i in range(0, len(ds), 37):
        context, _ = ds[i]
        for key, tensor in context.items():
            assert tensor.shape[0] == 512, (
                f"{key} at example {i} has shape {tensor.shape}"
            )


@pytest.mark.parametrize("stride,expected", [
    (1, 512),    # dense: one example per valid target
    (32, 16),    # (1024-512)/32 = 16
    (64, 8),     # (1024-512)/64 = 8
    (128, 4),    # (1024-512)/128 = 4
    (256, 2),    # (1024-512)/256 = 2
])
def test_trailing_dataset_stride_variants(stride, expected):
    """Different strides should produce the correct number of examples."""
    n = 1024
    features = _make_features(n)
    ds = TrailingWindowDataset(features, [n], window_size=512, stride=stride)
    assert len(ds) == expected, f"Stride {stride}: expected {expected}, got {len(ds)}"


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
