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
    build_process_name_vocab,
    map_process_name,
    bucket_return_value,
    bucket_args_num,
    TrailingWindowDataset,
    TARGET_FIELDS,
    chunk_contiguous_blocks,
    carve_attack_val,
    tokenize_texts,
    preprocess_features,
    load_next_event_pipeline,
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
    return preprocess_features(
        sample_df,
        process_name_vocab=build_process_name_vocab(sample_df["processName"]),
        args_vocab=build_vocab(sample_df["args"], 200),
        cat_vocabs={
            "userId": build_categorical_vocab(sample_df["userId"]),
            "eventId": build_categorical_vocab(sample_df["eventId"]),
        },
    )


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
    """preprocess_features should return exactly the seven input features."""
    features = _preprocess(sample_df)
    expected_keys = {
        "processName", "args_ids",
        "userId", "eventId",
        "argsNum", "returnValue", "parentProcessId",
    }
    assert set(features.keys()) == expected_keys


def test_preprocess_features_all_int64(sample_df):
    """Every feature array must be int64 class indices (no floats anywhere)."""
    features = _preprocess(sample_df)
    for key, arr in features.items():
        assert arr.dtype == np.int64, f"{key} has dtype {arr.dtype}"


def test_preprocess_features_buckets_in_range(sample_df):
    """argsNum/returnValue features must hold bucket classes, not raw values."""
    features = _preprocess(sample_df)
    assert features["argsNum"].min() >= 0 and features["argsNum"].max() <= 15
    assert features["returnValue"].min() >= 0 and features["returnValue"].max() <= 12


def test_preprocess_features_parent_pid_three_levels(sample_df):
    """parentProcessId must collapse to the closed 3-level index."""
    features = _preprocess(sample_df)
    assert set(np.unique(features["parentProcessId"])) <= {0, 1, 2}


def test_preprocess_features_bucketing_matches_bucketizers(sample_df):
    """argsNum/returnValue features must equal the standalone bucketizers."""
    features = _preprocess(sample_df)
    np.testing.assert_array_equal(
        features["returnValue"], bucket_return_value(sample_df["returnValue"])
    )
    np.testing.assert_array_equal(
        features["argsNum"], bucket_args_num(sample_df["argsNum"])
    )


def test_preprocess_features_real_zero_keeps_own_index():
    """With a vocab provided, userId 0 must NOT map to OOV."""
    df = pd.DataFrame({
        "userId": [0, 0, 1, 2],
        "eventId": [5, 5, 6, 7],
        "processName": ["bash"] * 4,
        "args": ["-c"] * 4,
        "argsNum": [0] * 4,
        "returnValue": [0] * 4,
        "parentProcessId": [0] * 4,
    })
    cat_vocabs = {
        "userId": build_categorical_vocab(df["userId"]),
        "eventId": build_categorical_vocab(df["eventId"]),
    }
    features = preprocess_features(
        df,
        process_name_vocab=build_process_name_vocab(df["processName"]),
        args_vocab=build_vocab(df["args"], 10),
        cat_vocabs=cat_vocabs,
    )
    assert (features["userId"] != 0).all(), "Real userId 0 mapped to OOV"


def test_preprocess_features_unseen_maps_to_oov():
    """Values unseen at vocab-build time must map to OOV index 0."""
    train_df = pd.DataFrame({
        "userId": [0, 0, 1],
        "eventId": [5, 5, 6],
        "processName": ["bash", "bash", "sshd"],
        "args": ["-c"] * 3,
        "argsNum": [0] * 3,
        "returnValue": [0] * 3,
        "parentProcessId": [0] * 3,
    })
    test_df = pd.DataFrame({
        "userId": [99],          # unseen user
        "eventId": [77],         # unseen eventId
        "processName": ["nc"],   # unseen processName
        "args": ["-c"],
        "argsNum": [0],
        "returnValue": [0],
        "parentProcessId": [0],
    })
    cat_vocabs = {
        "userId": build_categorical_vocab(train_df["userId"]),
        "eventId": build_categorical_vocab(train_df["eventId"]),
    }
    features = preprocess_features(
        test_df,
        process_name_vocab=build_process_name_vocab(train_df["processName"]),
        args_vocab=build_vocab(train_df["args"], 10),
        cat_vocabs=cat_vocabs,
    )
    assert features["userId"][0] == 0
    assert features["eventId"][0] == 0
    assert features["processName"][0] == 0


def test_preprocess_features_deterministic(sample_df):
    """Same df + same vocabs must produce identical features."""
    f1 = _preprocess(sample_df)
    f2 = _preprocess(sample_df)
    for key in f1:
        np.testing.assert_array_equal(f1[key], f2[key])


# ── Process-Name Vocabulary ────────────────────────────────────────────────────

def test_build_process_name_vocab_exact_strings():
    """processName vocab maps whole names, not sub-token fragments."""
    series = pd.Series(["systemd-udevd", "systemd-udevd", "sshd", "bash"])
    vocab = build_process_name_vocab(series)
    assert "systemd-udevd" in vocab
    assert "sshd" in vocab
    assert "<PAD>" not in vocab  # exact-match vocab, not token vocab


def test_build_process_name_vocab_zero_is_oov():
    """Index 0 must be reserved for OOV; real names start at 1."""
    series = pd.Series(["bash", "bash", "sshd"])
    vocab = build_process_name_vocab(series)
    assert min(vocab.values()) == 1


def test_map_process_name_unseen_is_oov():
    """Names absent from the vocab must map to 0."""
    vocab = build_process_name_vocab(pd.Series(["bash", "bash", "sshd"]))
    mapped = map_process_name(pd.Series(["bash", "nc", None]), vocab)
    assert mapped[0] != 0
    assert mapped[1] == 0
    assert mapped[2] == 0


def test_build_process_name_vocab_deterministic():
    """Ties must break consistently across calls."""
    series = pd.Series(["b", "b", "a", "a", "c"])
    v1 = build_process_name_vocab(series)
    v2 = build_process_name_vocab(series)
    assert v1 == v2


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


# ═══════════════════════════════════════════════════════════════════════════════
# Attack-Val Carve-Out Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_carve_df():
    """Two evil hosts (small contiguous evil bursts) + one benign test host."""
    frames = []
    ts = 0.0
    spec = [
        ("evil-a", 2000, (500, 540)),   # evil burst of 40 inside 2000 events
        ("benign-1", 1500, None),
        ("evil-b", 800, (300, 315)),    # evil burst of 15 inside 800 events
    ]
    for host, n, evil_range in spec:
        evil = np.zeros(n, dtype=np.int64)
        if evil_range is not None:
            evil[evil_range[0]:evil_range[1]] = 1
        frames.append(pd.DataFrame({
            "timestamp": np.arange(ts, ts + n, dtype=np.float64),
            "processId": np.arange(n) % 1000,
            "parentProcessId": np.ones(n, dtype=np.int64),
            "userId": np.zeros(n, dtype=np.int64),
            "processName": ["bash"] * n,
            "hostName": [host] * n,
            "eventId": np.ones(n, dtype=np.int64),
            "eventName": ["execve"] * n,
            "argsNum": np.ones(n, dtype=np.int64),
            "returnValue": np.zeros(n, dtype=np.int64),
            "args": ["-c"] * n,
            "sus": np.zeros(n, dtype=np.int64),
            "evil": evil,
        }))
        ts += n + 0.5
    return pd.concat(frames, ignore_index=True)


# ── Contiguous Blocks ──────────────────────────────────────────────────────────

def test_chunk_blocks_tile_stream():
    """Blocks must cover [0, n) with no gaps or overlaps."""
    blocks = chunk_contiguous_blocks(1003, 50)
    assert blocks[0][0] == 0
    assert blocks[-1][1] == 1003
    for (s1, e1), (s2, e2) in zip(blocks, blocks[1:]):
        assert e1 == s2, f"Gap/overlap between ({s1},{e1}) and ({s2},{e2})"


def test_chunk_blocks_fewer_events_than_blocks():
    """n < n_blocks must degrade to single-event blocks, still tiling."""
    blocks = chunk_contiguous_blocks(5, 50)
    assert len(blocks) == 5
    assert blocks[0][0] == 0 and blocks[-1][1] == 5


# ── Carve-Out Behaviour ────────────────────────────────────────────────────────

def test_carve_disjoint_and_complete():
    """Tune and test must be disjoint and together preserve every row."""
    df = _make_carve_df()
    tune, test = carve_attack_val(df, seed=42)

    assert len(tune) + len(test) == len(df)
    tune_keys = set(zip(tune["hostName"], tune["timestamp"]))
    test_keys = set(zip(test["hostName"], test["timestamp"]))
    assert tune_keys.isdisjoint(test_keys), "A row landed in both tune and test"


def test_carve_tune_has_evil_from_both_evil_hosts():
    """The tuning set must contain evil events from every evil test host."""
    df = _make_carve_df()
    tune, _ = carve_attack_val(df, seed=7, tune_frac=0.05)  # tiny frac forces fix-up

    for evil_host in ["evil-a", "evil-b"]:
        host_evil = tune[(tune["hostName"] == evil_host) & (tune["evil"] == 1)]
        assert len(host_evil) > 0, f"No evil from {evil_host} in tune set"


def test_carve_tune_has_benign_from_benign_hosts():
    """The tuning set must contain benign events from benign test hosts."""
    df = _make_carve_df()
    tune, _ = carve_attack_val(df, seed=0, tune_frac=0.01)  # extreme: ~no blocks drawn

    host_benign = tune[(tune["hostName"] == "benign-1") & (tune["evil"] == 0)]
    assert len(host_benign) > 0, "No benign events from benign-1 in tune set"


def test_carve_never_splits_a_block():
    """Every contiguous block must go wholly to tune or wholly to test."""
    df = _make_carve_df()
    tune, test = carve_attack_val(df, seed=42, n_blocks=25)

    for host in df["hostName"].unique():
        host_df = df[df["hostName"] == host].sort_values("timestamp").reset_index(drop=True)
        tune_positions = host_df.index[
            host_df.set_index(["hostName", "timestamp"]).index.isin(
                set(zip(tune["hostName"], tune["timestamp"]))
            )
        ]
        tune_pos = set(tune_positions)
        for s, e in chunk_contiguous_blocks(len(host_df), 25):
            in_tune = tune_pos.intersection(range(s, e))
            assert len(in_tune) == 0 or len(in_tune) == e - s, (
                f"Block ({s},{e}) of {host} was split between tune and test"
            )


def test_carve_deterministic_under_seed():
    """Same seed must produce identical tune/test partitions."""
    df = _make_carve_df()
    tune1, test1 = carve_attack_val(df, seed=123)
    tune2, test2 = carve_attack_val(df, seed=123)
    pd.testing.assert_frame_equal(tune1, tune2)
    pd.testing.assert_frame_equal(test1, test2)


def test_carve_different_seeds_differ():
    """Different seeds should (overwhelmingly) produce different partitions."""
    df = _make_carve_df()
    tune1, _ = carve_attack_val(df, seed=1)
    tune2, _ = carve_attack_val(df, seed=2)
    assert not tune1.equals(tune2)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_host_df(host, n, evil_slice=None, ts_start=0.0):
    """One host's synthetic BETH-style frame (columns match per-host CSVs)."""
    evil = np.zeros(n, dtype=np.int64)
    if evil_slice is not None:
        evil[evil_slice[0]:evil_slice[1]] = 1
    rng = np.random.default_rng(abs(hash(host)) % 2**31)
    return pd.DataFrame({
        "timestamp": np.arange(ts_start, ts_start + n, dtype=np.float64),
        "processId": rng.integers(1, 1000, n),
        "parentProcessId": rng.integers(0, 3, n),
        "userId": rng.choice([0, 0, 0, 1], n),
        "processName": rng.choice(["systemd", "bash", "curl"], n),
        "hostName": [host] * n,
        "eventId": rng.integers(1, 20, n),
        "eventName": rng.choice(["execve", "open"], n),
        "argsNum": rng.integers(0, 5, n),
        "returnValue": rng.choice([0, 0, -1, 3], n),
        "args": rng.choice(["-c ls", "", "-l"], n),
        "sus": np.zeros(n, dtype=np.int64),
        "evil": evil,
    })


@pytest.fixture
def pipeline_csv_dir(tmp_path):
    """Three benign hosts + two evil hosts (small evil bursts), as CSVs."""
    per_host = tmp_path / "per_host"
    per_host.mkdir()
    ts = 0.0
    for host, evil_slice in [
        ("benign-a", None),
        ("benign-b", None),
        ("benign-c", None),
        ("evil-1", (100, 130)),
        ("evil-2", (300, 320)),
    ]:
        df = _make_host_df(host, 600, evil_slice, ts_start=ts)
        ts += 600 + 1.0
        df.to_csv(per_host / f"{host}.csv", index=False)
    return str(per_host)


def test_load_next_event_pipeline_returns_four_datasets(pipeline_csv_dir):
    """Pipeline must return train/val (benign), tune (carved), and test."""
    train_ds, val_ds, tune_ds, test_ds, vocab_sizes, vocabs = \
        load_next_event_pipeline(pipeline_csv_dir, window_size=128)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(tune_ds) > 0
    assert len(test_ds) > 0


def test_load_next_event_pipeline_split_discipline(pipeline_csv_dir):
    """Train/val must be benign-only; test must hold the evil hosts."""
    train_ds, val_ds, tune_ds, test_ds, _, _ = \
        load_next_event_pipeline(pipeline_csv_dir, window_size=128)

    assert (train_ds.labels == 1).sum() == 0, "Training data must be benign-only"
    assert (val_ds.labels == 1).sum() == 0, "Validation data must be benign-only"
    assert (test_ds.labels == 1).sum() > 0, "Test set must contain evil events"


def test_load_next_event_pipeline_tune_covers_both_evil_hosts(pipeline_csv_dir):
    """The carved tuning frame must contain evil events from both evil hosts.

    Frame-level guarantee (the carve-out's contract). Scored evil also
    depends on trailing-context availability inside each carve segment,
    which tiny fixtures cannot promise.
    """
    _, _, tune_ds, _, _, _ = load_next_event_pipeline(
        pipeline_csv_dir, window_size=128, seed=7,
    )

    # Labels span the full carved frame; hosts are contiguous segments in
    # host_lengths order.
    evil_per_host = []
    offset = 0
    for host_n in tune_ds.host_lengths:
        evil_per_host.append(int(tune_ds.labels[offset:offset + host_n].sum()))
        offset += host_n

    n_hosts_with_evil = sum(1 for count in evil_per_host if count > 0)
    assert n_hosts_with_evil >= 2, (
        f"Expected evil from both evil hosts, per-host counts: {evil_per_host}"
    )


def test_load_next_event_pipeline_vocab_sizes(pipeline_csv_dir):
    """Vocab sizes must account for OOV and match the fixed bucket schemas."""
    _, _, _, _, vocab_sizes, vocabs = \
        load_next_event_pipeline(pipeline_csv_dir, window_size=128)

    # userId values {0, 1} → 2 real + 1 OOV
    assert vocab_sizes["user_id_vocab_size"] == len(vocabs["cat"]["userId"]) + 1
    assert vocab_sizes["user_id_vocab_size"] >= 2
    # Fixed bucket schemas
    assert vocab_sizes["args_num_vocab_size"] == 16
    assert vocab_sizes["return_value_vocab_size"] == 13
    assert vocab_sizes["parent_pid_vocab_size"] == 3


def test_load_next_event_pipeline_examples_are_wellformed(pipeline_csv_dir):
    """A training example must have full-length context and 5 targets."""
    train_ds, _, _, _, _, _ = load_next_event_pipeline(
        pipeline_csv_dir, window_size=128, train_stride=4,
    )
    context, targets = train_ds[0]

    for key, tensor in context.items():
        assert tensor.shape[0] == 128, f"{key} context shape {tensor.shape}"
        assert tensor.dtype == torch.int64
    assert set(targets.keys()) == set(TARGET_FIELDS)
    assert targets["argsNum"].item() <= 15
    assert targets["returnValue"].item() <= 12


def test_load_next_event_pipeline_deterministic(pipeline_csv_dir):
    """Same seed must yield identical datasets."""
    p1 = load_next_event_pipeline(pipeline_csv_dir, window_size=128, seed=99)
    p2 = load_next_event_pipeline(pipeline_csv_dir, window_size=128, seed=99)

    for ds1, ds2 in zip(p1[:4], p2[:4]):
        assert len(ds1) == len(ds2)
        assert ds1.centers == ds2.centers
        np.testing.assert_array_equal(ds1.labels, ds2.labels)
    assert p1[4] == p2[4]


def test_load_next_event_pipeline_missing_dir_raises(tmp_path):
    """Missing data directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_next_event_pipeline(str(tmp_path / "does_not_exist"))
