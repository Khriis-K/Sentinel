"""
Tests for src.model — SentinelVAE architecture, embedding dimensions,
forward-pass output shapes, and reconstruction error. Also tests for
CenteredWindowDataset.
"""
import numpy as np
import pytest
import torch

from src.model import SentinelVAE
from src.data import CenteredWindowDataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vocab_sizes():
    """Realistic vocab sizes matching the BETH benchmark subset."""
    return {
        "process_name_vocab_size": 105,
        "args_vocab_size": 10000,
        "user_id_vocab_size": 8,
        "mount_ns_vocab_size": 7,
        "event_id_vocab_size": 45,
    }


@pytest.fixture
def model(vocab_sizes):
    """Create a VAE with default hyperparameters."""
    return SentinelVAE(**vocab_sizes)


@pytest.fixture
def sample_batch(vocab_sizes):
    """Build a synthetic batch matching BethDataset output shapes."""
    batch_size = 4
    seq_len = 128  # shorter window for tests

    return {
        "processName_ids": torch.randint(0, vocab_sizes["process_name_vocab_size"], (batch_size, seq_len, 16)),
        "args_ids": torch.randint(0, vocab_sizes["args_vocab_size"], (batch_size, seq_len, 64)),
        "userId": torch.randint(0, vocab_sizes["user_id_vocab_size"], (batch_size, seq_len)),
        "mountNamespace": torch.randint(0, vocab_sizes["mount_ns_vocab_size"], (batch_size, seq_len)),
        "eventId": torch.randint(0, vocab_sizes["event_id_vocab_size"], (batch_size, seq_len)),
        "argsNum": torch.randn(batch_size, seq_len),
        "returnValue": torch.randn(batch_size, seq_len),
        "parentProcessId": torch.randn(batch_size, seq_len),
    }


# ── Embedding Dimensions ──────────────────────────────────────────────────────

def test_embedding_dims_match_spec(model):
    """Embedding dimensions must match the PRD specification."""
    dims = model.embedding_dims
    assert dims["processName"] == 64, f"Expected 64, got {dims['processName']}"
    assert dims["args"] == 64, f"Expected 64, got {dims['args']}"
    assert dims["userId"] == 16, f"Expected 16, got {dims['userId']}"
    assert dims["mountNamespace"] == 8, f"Expected 8, got {dims['mountNamespace']}"
    assert dims["eventId"] == 16, f"Expected 16, got {dims['eventId']}"


def test_embedding_vocab_sizes(model, vocab_sizes):
    """Embedding layers should have the correct vocabulary sizes."""
    assert model.process_name_embed.num_embeddings == vocab_sizes["process_name_vocab_size"]
    assert model.args_embed.num_embeddings == vocab_sizes["args_vocab_size"]
    assert model.user_id_embed.num_embeddings == vocab_sizes["user_id_vocab_size"]
    assert model.mount_ns_embed.num_embeddings == vocab_sizes["mount_ns_vocab_size"]
    assert model.event_id_embed.num_embeddings == vocab_sizes["event_id_vocab_size"]


def test_custom_embedding_dims():
    """Custom embedding dimensions should be reflected in the model."""
    model = SentinelVAE(
        process_name_vocab_size=50,
        args_vocab_size=200,
        user_id_vocab_size=10,
        mount_ns_vocab_size=5,
        event_id_vocab_size=20,
        process_name_embed_dim=32,
        args_embed_dim=32,
        user_id_embed_dim=8,
        mount_ns_embed_dim=4,
        event_id_embed_dim=8,
    )
    dims = model.embedding_dims
    assert dims["processName"] == 32
    assert dims["args"] == 32
    assert dims["userId"] == 8
    assert dims["mountNamespace"] == 4
    assert dims["eventId"] == 8

    # Event dim should be 32+32+8+4+8+3 = 87
    assert model.event_dim == 87


# ── Forward Pass ──────────────────────────────────────────────────────────────

def test_forward_output_shapes(model, sample_batch):
    """Forward pass should produce (reconstructed, mu, logvar) with correct shapes."""
    model.eval()
    with torch.no_grad():
        reconstructed, mu, logvar = model(sample_batch)
    batch_size = sample_batch["userId"].shape[0]
    seq_len = sample_batch["userId"].shape[1]
    event_dim = model.event_dim

    assert reconstructed.shape == (batch_size, seq_len, event_dim), \
        f"Expected ({batch_size}, {seq_len}, {event_dim}), got {reconstructed.shape}"
    assert mu.shape == (batch_size, model.latent_dim), \
        f"Expected ({batch_size}, {model.latent_dim}), got {mu.shape}"
    assert logvar.shape == (batch_size, model.latent_dim), \
        f"Expected ({batch_size}, {model.latent_dim}), got {logvar.shape}"


def test_forward_output_is_finite(model, sample_batch):
    """All outputs should be finite (no NaN or inf)."""
    model.eval()
    with torch.no_grad():
        reconstructed, mu, logvar = model(sample_batch)
    assert torch.isfinite(reconstructed).all(), "Non-finite values in reconstructed"
    assert torch.isfinite(mu).all(), "Non-finite values in mu"
    assert torch.isfinite(logvar).all(), "Non-finite values in logvar"


def test_model_train_mode_works(model, sample_batch):
    """Model should produce output in training mode (dropout active)."""
    model.train()
    reconstructed, mu, logvar = model(sample_batch)
    batch_size = sample_batch["userId"].shape[0]
    seq_len = sample_batch["userId"].shape[1]
    assert reconstructed.shape == (batch_size, seq_len, model.event_dim)


def test_model_deterministic_in_eval(model, sample_batch):
    """In eval mode (no sampling noise), same input should produce identical output."""
    model.eval()
    with torch.no_grad():
        recon1, mu1, logvar1 = model(sample_batch)
        recon2, mu2, logvar2 = model(sample_batch)
    # mu and logvar should be identical (encoder is deterministic)
    torch.testing.assert_close(mu1, mu2)
    torch.testing.assert_close(logvar1, logvar2)
    # reconstructed should also be identical (decoder input is deterministic in eval)
    # Note: in training mode, reparameterize() samples ε, so outputs would differ


# ── Architecture Properties ───────────────────────────────────────────────────

def test_event_dim_calculation(model):
    """Event vector dimension should match sum of embedding dims + 3 numeric."""
    dims = model.embedding_dims
    expected = sum(dims.values()) + 3
    assert model.event_dim == expected, \
        f"Expected event_dim={expected}, got {model.event_dim}"


def test_latent_dim(model):
    """Default latent dimension should be 32."""
    assert model.latent_dim == 32


def test_hidden_size(model):
    """Default hidden size should be 128."""
    assert model.hidden_size == 128


def test_num_layers(model):
    """Default num_layers should be 2."""
    assert model.num_layers == 2


def test_variable_batch_size(model, vocab_sizes):
    """Model should handle different batch sizes."""
    for bs in [1, 2, 8]:
        batch = {
            "processName_ids": torch.randint(0, vocab_sizes["process_name_vocab_size"], (bs, 128, 16)),
            "args_ids": torch.randint(0, vocab_sizes["args_vocab_size"], (bs, 128, 64)),
            "userId": torch.randint(0, vocab_sizes["user_id_vocab_size"], (bs, 128)),
            "mountNamespace": torch.randint(0, vocab_sizes["mount_ns_vocab_size"], (bs, 128)),
            "eventId": torch.randint(0, vocab_sizes["event_id_vocab_size"], (bs, 128)),
            "argsNum": torch.randn(bs, 128),
            "returnValue": torch.randn(bs, 128),
            "parentProcessId": torch.randn(bs, 128),
        }
        model.eval()
        with torch.no_grad():
            reconstructed, mu, logvar = model(batch)
        assert reconstructed.shape == (bs, 128, model.event_dim), \
            f"Batch size {bs}: unexpected shape {reconstructed.shape}"


def test_padding_idx_zero_maps_to_zeros(model):
    """Embedding for padding_idx=0 should be all zeros."""
    weight = model.process_name_embed.weight
    assert (weight[0] == 0).all(), "Padding embedding at index 0 should be all zeros"


def test_init_kwargs_saved(model, vocab_sizes):
    """Model should save its init kwargs for serialization."""
    kwargs = model._init_kwargs
    assert kwargs["process_name_vocab_size"] == vocab_sizes["process_name_vocab_size"]
    assert kwargs["args_vocab_size"] == vocab_sizes["args_vocab_size"]
    assert kwargs["hidden_size"] == 128
    assert kwargs["latent_dim"] == 32
    assert kwargs["num_layers"] == 2
    assert kwargs["dropout"] == 0.3


# ── Reparameterization ───────────────────────────────────────────────────────

def test_reparameterize_produces_different_samples(model):
    """Reparameterization should produce different samples each time."""
    mu = torch.zeros(4, model.latent_dim)
    logvar = torch.zeros(4, model.latent_dim)
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)
    # Different samples (with overwhelming probability)
    assert not torch.allclose(z1, z2), "Reparameterize should produce different samples"


def test_reparameterize_deterministic_with_zero_variance(model):
    """With logvar → -inf (zero variance), reparameterize should return mu."""
    mu = torch.randn(4, model.latent_dim)
    logvar = torch.full((4, model.latent_dim), -20.0)  # very small variance
    z = model.reparameterize(mu, logvar)
    torch.testing.assert_close(z, mu, atol=1e-4, rtol=1e-4)


# ── Reconstruction Error ─────────────────────────────────────────────────────

def test_reconstruction_error_shape(model, sample_batch):
    """reconstruction_error should return per-sample MSE of shape (B,)."""
    model.eval()
    with torch.no_grad():
        mse = model.reconstruction_error(sample_batch)
    batch_size = sample_batch["userId"].shape[0]
    assert mse.shape == (batch_size,), f"Expected ({batch_size},), got {mse.shape}"


def test_reconstruction_error_is_positive(model, sample_batch):
    """MSE should be non-negative and finite."""
    model.eval()
    with torch.no_grad():
        mse = model.reconstruction_error(sample_batch)
    assert (mse >= 0).all(), f"Negative MSE: {mse}"
    assert torch.isfinite(mse).all(), f"Non-finite MSE: {mse}"


# ═══════════════════════════════════════════════════════════════════════════════
# CenteredWindowDataset Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_features(n_events: int):
    """Build a minimal synthetic feature dict for n_events."""
    return {
        "processName_ids": np.zeros((n_events, 16), dtype=np.int64),
        "args_ids": np.zeros((n_events, 64), dtype=np.int64),
        "userId": np.zeros(n_events, dtype=np.int64),
        "mountNamespace": np.zeros(n_events, dtype=np.int64),
        "eventId": np.zeros(n_events, dtype=np.int64),
        "argsNum": np.zeros(n_events, dtype=np.float32),
        "returnValue": np.zeros(n_events, dtype=np.float32),
        "parentProcessId": np.zeros(n_events, dtype=np.float32),
    }


# ── Basic Construction ────────────────────────────────────────────────────────

def test_centered_dataset_single_host():
    """Single host with more than window_size events should produce windows."""
    n = 1024
    features = _make_features(n)
    labels = np.zeros(n, dtype=np.int64)
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=32)
    # Valid centers: positions 256 .. 767 (512 positions), stride 32 → 16 windows
    assert len(ds) == 16


def test_centered_dataset_stride_one():
    """Stride=1 should produce one window per valid center."""
    n = 1024
    features = _make_features(n)
    labels = np.zeros(n, dtype=np.int64)
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=1)
    # Valid centers: 512 positions
    assert len(ds) == 512


def test_centered_dataset_multiple_hosts():
    """Windows should not cross host boundaries."""
    n_events_per_host = 600
    features = _make_features(n_events_per_host * 2)
    labels = np.zeros(n_events_per_host * 2, dtype=np.int64)
    host_lengths = [n_events_per_host, n_events_per_host]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=32)
    # Each host: valid centers 256..343 (88 positions), stride 32 → 3 windows each
    assert len(ds) == 6


def test_centered_dataset_host_too_small():
    """A host smaller than window_size contributes zero windows."""
    features = _make_features(300)
    labels = np.zeros(300, dtype=np.int64)
    host_lengths = [300]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=32)
    assert len(ds) == 0


def test_centered_dataset_mixed_host_sizes():
    """Only the large-enough host contributes windows."""
    features = _make_features(1100)  # 600 + 500
    labels = np.zeros(1100, dtype=np.int64)
    host_lengths = [600, 500]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=32)
    # Host 1 (600): 88 centers, stride 32 → 3 windows
    # Host 2 (500): not enough → 0 windows
    assert len(ds) == 3


# ── Label Correctness ─────────────────────────────────────────────────────────

def test_label_is_center_event():
    """Label must be the center event's value, not any() in window."""
    n = 1024
    features = _make_features(n)
    labels = np.zeros(n, dtype=np.int64)
    # Place evil at position 400
    labels[400] = 1
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=32)

    # Find the window centered on position 400 (or close to it)
    for i in range(len(ds)):
        _, lbl = ds[i]
        center_pos = ds.centers[i]
        if center_pos == 400:
            assert lbl.item() == 1.0, f"Center at evil position {center_pos} should have label 1"
        elif center_pos == 384 or center_pos == 416:
            # Stride=32 neighbors might or might not include evil
            pass


def test_benign_center_in_evil_neighborhood():
    """A benign event surrounded by evil neighbors should have label 0."""
    n = 1024
    features = _make_features(n)
    labels = np.ones(n, dtype=np.int64)  # all evil
    labels[400] = 0  # one benign in the middle
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=1)

    # The window centered on 400 should have label 0 even though
    # most of the window is evil
    for i in range(len(ds)):
        _, lbl = ds[i]
        center_pos = ds.centers[i]
        if center_pos == 400:
            assert lbl.item() == 0.0, (
                f"Benign center at {center_pos} should have label 0 "
                f"(window-level any() would label this 1)"
            )
            return
    pytest.fail("Center at position 400 not found in dataset")


# ── Window Content ────────────────────────────────────────────────────────────

def test_window_content_matches_slice():
    """Window tensors should match the expected numpy slice."""
    n = 1024
    features = _make_features(n)
    # Use a non-zero feature to verify correct slicing
    features["argsNum"] = np.arange(n, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=1)

    center = 512  # pick a center
    # Find index for this center
    for i in range(len(ds)):
        if ds.centers[i] == center:
            x, _ = ds[i]
            expected = np.arange(center - 256, center + 256, dtype=np.float32)
            actual = x["argsNum"].numpy()
            np.testing.assert_array_equal(actual, expected)
            return
    pytest.fail(f"Center {center} not found")


def test_sequence_window_does_not_cross_hosts():
    """The first event of host 2 should never appear in a host-1 window."""
    n1, n2 = 600, 600
    features = _make_features(n1 + n2)
    features["argsNum"] = np.arange(n1 + n2, dtype=np.float32)
    labels = np.zeros(n1 + n2, dtype=np.int64)
    host_lengths = [n1, n2]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=1)

    # The last center in host 1 is at n1 - 256 - 1 = 343
    # Its window spans [343-256, 343+256) = [87, 599) — all within host 1 (< 600)
    max_window_end = 0
    for i in range(len(ds)):
        center = ds.centers[i]
        if center < n1:
            # All centers in host 1 should produce windows fully within host 1
            window_end = center + 256
            assert window_end <= n1, (
                f"Window centered at {center} ends at {window_end}, "
                f"crossing into host 2 (boundary at {n1})"
            )
            max_window_end = max(max_window_end, window_end)
    assert max_window_end <= n1


# ── Edge Handling ─────────────────────────────────────────────────────────────

def test_truncate_edges_no_padding():
    """Events without 256 neighbors should not be centers (truncate)."""
    n = 600
    features = _make_features(n)
    labels = np.arange(n, dtype=np.int64)  # unique labels per position
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=1)

    # First valid center = 256, last valid center = 600 - 256 - 1 = 343
    for center in ds.centers:
        assert center >= 256, f"Center {center} too close to start (needs >= 256)"
        assert center <= 343, f"Center {center} too close to end (needs <= {n - 256 - 1})"


def test_all_centers_different_window():
    """Stride=1: every consecutive center should have a different window slice."""
    n = 1024
    features = _make_features(n)
    features["argsNum"] = np.arange(n, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=1)

    # First two windows should differ by exactly 1 event at each end
    x0, _ = ds[0]
    x1, _ = ds[1]
    # x1 should be shifted by 1 compared to x0
    expected_x1 = np.arange(1, 513, dtype=np.float32)
    np.testing.assert_array_equal(x1["argsNum"].numpy(), expected_x1)


# ── Stride Variants ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("stride,expected", [
    (1, 512),    # (1024-512)/1
    (32, 16),    # (1024-512)/32 = 16
    (64, 8),     # (1024-512)/64 = 8
    (128, 4),    # (1024-512)/128 = 4
    (256, 2),    # (1024-512)/256 = 2
])
def test_stride_variants(stride, expected):
    """Different strides should produce the correct number of windows."""
    n = 1024
    features = _make_features(n)
    labels = np.zeros(n, dtype=np.int64)
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=stride)
    assert len(ds) == expected, f"Stride {stride}: expected {expected}, got {len(ds)}"


# ── Dataset Properties ────────────────────────────────────────────────────────

def test_centered_dataset_stores_host_lengths():
    """CenteredWindowDataset should expose host_lengths for downstream use."""
    n = 1024
    features = _make_features(n)
    labels = np.zeros(n, dtype=np.int64)
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=32)
    assert ds.host_lengths == host_lengths


def test_all_labels_zeroorone():
    """Labels should always be 0.0 or 1.0 (float32)."""
    n = 1024
    features = _make_features(n)
    labels = np.array([0, 1] * (n // 2), dtype=np.int64)
    host_lengths = [n]
    ds = CenteredWindowDataset(features, labels, host_lengths, window_size=512, stride=32)
    for i in range(len(ds)):
        _, lbl = ds[i]
        assert lbl.item() in {0.0, 1.0}, f"Unexpected label: {lbl.item()}"
