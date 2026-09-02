"""
Tests for src.model — NextEventLSTM architecture, embedding dimensions,
forward-pass output shapes, per-field logits, and surprisal scoring.
"""
import numpy as np
import pytest
import torch

from src.model import NextEventLSTM


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vocab_sizes():
    """Vocab sizes matching the BETH benchmark (each includes the OOV slot)."""
    return {
        "process_name_vocab_size": 105,
        "args_vocab_size": 10000,
        "user_id_vocab_size": 8,
        "event_id_vocab_size": 45,
        "args_num_vocab_size": 16,
        "return_value_vocab_size": 13,
        "parent_pid_vocab_size": 3,
    }


@pytest.fixture
def model(vocab_sizes):
    """Create a next-event model with default hyperparameters."""
    return NextEventLSTM(**vocab_sizes)


@pytest.fixture
def sample_context(vocab_sizes):
    """Synthetic context batch matching TrailingWindowDataset output shapes."""
    batch_size = 4
    seq_len = 128  # shorter window for tests

    return {
        "processName": torch.randint(0, vocab_sizes["process_name_vocab_size"], (batch_size, seq_len)),
        "args_ids": torch.randint(0, vocab_sizes["args_vocab_size"], (batch_size, seq_len, 64)),
        "userId": torch.randint(0, vocab_sizes["user_id_vocab_size"], (batch_size, seq_len)),
        "eventId": torch.randint(0, vocab_sizes["event_id_vocab_size"], (batch_size, seq_len)),
        "argsNum": torch.randint(0, vocab_sizes["args_num_vocab_size"], (batch_size, seq_len)),
        "returnValue": torch.randint(0, vocab_sizes["return_value_vocab_size"], (batch_size, seq_len)),
        "parentProcessId": torch.randint(0, vocab_sizes["parent_pid_vocab_size"], (batch_size, seq_len)),
    }


@pytest.fixture
def sample_targets(vocab_sizes):
    """Synthetic targets for the five predicted fields."""
    batch_size = 4
    return {
        "eventId": torch.randint(0, vocab_sizes["event_id_vocab_size"], (batch_size,)),
        "processName": torch.randint(0, vocab_sizes["process_name_vocab_size"], (batch_size,)),
        "userId": torch.randint(0, vocab_sizes["user_id_vocab_size"], (batch_size,)),
        "returnValue": torch.randint(0, vocab_sizes["return_value_vocab_size"], (batch_size,)),
        "argsNum": torch.randint(0, vocab_sizes["args_num_vocab_size"], (batch_size,)),
    }


# ── Embedding Dimensions ──────────────────────────────────────────────────────

def test_embedding_dims_match_defaults(model):
    """Default embedding dims carried over from the VAE-era stack."""
    dims = model.embedding_dims
    assert dims["processName"] == 64, f"Expected 64, got {dims['processName']}"
    assert dims["args"] == 64, f"Expected 64, got {dims['args']}"
    assert dims["userId"] == 16, f"Expected 16, got {dims['userId']}"
    assert dims["eventId"] == 16, f"Expected 16, got {dims['eventId']}"
    assert dims["argsNum"] == 8, f"Expected 8, got {dims['argsNum']}"
    assert dims["returnValue"] == 8, f"Expected 8, got {dims['returnValue']}"
    assert dims["parentProcessId"] == 4, f"Expected 4, got {dims['parentProcessId']}"


def test_embedding_vocab_sizes_wired_through(model, vocab_sizes):
    """Embedding tables must reflect the vocab sizes passed at construction."""
    assert model.process_name_embed.num_embeddings == vocab_sizes["process_name_vocab_size"]
    assert model.args_embed.num_embeddings == vocab_sizes["args_vocab_size"]
    assert model.user_id_embed.num_embeddings == vocab_sizes["user_id_vocab_size"]
    assert model.event_id_embed.num_embeddings == vocab_sizes["event_id_vocab_size"]
    assert model.args_num_embed.num_embeddings == vocab_sizes["args_num_vocab_size"]
    assert model.return_value_embed.num_embeddings == vocab_sizes["return_value_vocab_size"]
    assert model.parent_pid_embed.num_embeddings == vocab_sizes["parent_pid_vocab_size"]


def test_head_vocab_sizes_wired_through(model, vocab_sizes):
    """Each per-field head must output exactly vocab_size logits."""
    expected = {
        "processName": vocab_sizes["process_name_vocab_size"],
        "userId": vocab_sizes["user_id_vocab_size"],
        "eventId": vocab_sizes["event_id_vocab_size"],
        "returnValue": vocab_sizes["return_value_vocab_size"],
        "argsNum": vocab_sizes["args_num_vocab_size"],
    }
    for field, size in expected.items():
        assert model.heads[field].out_features == size, (
            f"Head {field} outputs {model.heads[field].out_features}, expected {size}"
        )


def test_custom_embedding_dims():
    """Custom embedding dimensions should be reflected in the model."""
    model = NextEventLSTM(
        process_name_vocab_size=50,
        args_vocab_size=200,
        user_id_vocab_size=10,
        event_id_vocab_size=20,
        args_num_vocab_size=16,
        return_value_vocab_size=13,
        parent_pid_vocab_size=3,
        process_name_embed_dim=32,
        args_embed_dim=32,
        user_id_embed_dim=8,
        event_id_embed_dim=8,
        args_num_embed_dim=4,
        return_value_embed_dim=4,
        parent_pid_embed_dim=2,
    )
    dims = model.embedding_dims
    assert dims["processName"] == 32
    assert dims["args"] == 32
    assert dims["userId"] == 8
    assert dims["eventId"] == 8
    assert dims["argsNum"] == 4
    assert dims["returnValue"] == 4
    assert dims["parentProcessId"] == 2

    # Event dim should be 32+32+8+8+4+4+2 = 90
    assert model.event_dim == 90


def test_event_dim_calculation(model):
    """Event vector dimension should equal the sum of all embedding dims."""
    dims = model.embedding_dims
    expected = sum(dims.values())
    assert model.event_dim == expected, \
        f"Expected event_dim={expected}, got {model.event_dim}"


def test_oov_embeddings_are_learnable(model):
    """Index 0 is OOV (a novelty signal), not padding — its embedding must train.

    Only the args token embedding may freeze index 0 (genuine PAD).
    """
    for name in ["process_name_embed", "user_id_embed", "event_id_embed"]:
        embed = getattr(model, name)
        assert embed.padding_idx is None, (
            f"{name} must not freeze index 0 (OOV is a real class)"
        )
    assert model.args_embed.padding_idx == 0, "args PAD index must stay frozen"


# ── Forward Pass ──────────────────────────────────────────────────────────────

def test_forward_output_shapes(model, sample_context, vocab_sizes):
    """Forward pass must produce per-field logits of shape (B, vocab_k)."""
    model.eval()
    with torch.no_grad():
        logits = model(sample_context)

    expected_fields = {"eventId", "processName", "userId", "returnValue", "argsNum"}
    assert set(logits.keys()) == expected_fields

    batch_size = sample_context["userId"].shape[0]
    expected_sizes = {
        "eventId": vocab_sizes["event_id_vocab_size"],
        "processName": vocab_sizes["process_name_vocab_size"],
        "userId": vocab_sizes["user_id_vocab_size"],
        "returnValue": vocab_sizes["return_value_vocab_size"],
        "argsNum": vocab_sizes["args_num_vocab_size"],
    }
    for field, size in expected_sizes.items():
        assert logits[field].shape == (batch_size, size), (
            f"logits[{field}] shape {logits[field].shape}, expected ({batch_size}, {size})"
        )


def test_forward_output_is_finite(model, sample_context):
    """All logits must be finite (no NaN or inf)."""
    model.eval()
    with torch.no_grad():
        logits = model(sample_context)
    for field, values in logits.items():
        assert torch.isfinite(values).all(), f"Non-finite logits for {field}"


def test_model_train_mode_works(model, sample_context):
    """Model should produce output in training mode (dropout active)."""
    model.train()
    logits = model(sample_context)
    batch_size = sample_context["userId"].shape[0]
    for field, values in logits.items():
        assert values.shape[0] == batch_size


def test_variable_batch_size(model, vocab_sizes):
    """Model should handle different batch sizes."""
    for bs in [1, 2, 8]:
        context = {
            "processName": torch.randint(0, vocab_sizes["process_name_vocab_size"], (bs, 128)),
            "args_ids": torch.randint(0, vocab_sizes["args_vocab_size"], (bs, 128, 64)),
            "userId": torch.randint(0, vocab_sizes["user_id_vocab_size"], (bs, 128)),
            "eventId": torch.randint(0, vocab_sizes["event_id_vocab_size"], (bs, 128)),
            "argsNum": torch.randint(0, vocab_sizes["args_num_vocab_size"], (bs, 128)),
            "returnValue": torch.randint(0, vocab_sizes["return_value_vocab_size"], (bs, 128)),
            "parentProcessId": torch.randint(0, vocab_sizes["parent_pid_vocab_size"], (bs, 128)),
        }
        model.eval()
        with torch.no_grad():
            logits = model(context)
        for field, values in logits.items():
            assert values.shape == (bs, model.heads[field].out_features), (
                f"Batch size {bs}, field {field}: unexpected shape {values.shape}"
            )


# ── Surprisal ─────────────────────────────────────────────────────────────────

def test_surprisal_shapes(model, sample_context, sample_targets):
    """surprisal must return per-field (B,) tensors plus a 'total' (B,) tensor."""
    model.eval()
    with torch.no_grad():
        surprisal = model.surprisal(sample_context, sample_targets)

    for field in ["eventId", "processName", "userId", "returnValue", "argsNum", "total"]:
        assert field in surprisal, f"Missing surprisal key {field}"
        assert surprisal[field].shape == (sample_targets["eventId"].shape[0],)


def test_surprisal_positive_and_finite(model, sample_context, sample_targets):
    """Surprisals are -log probabilities: finite and non-negative."""
    model.eval()
    with torch.no_grad():
        surprisal = model.surprisal(sample_context, sample_targets)
    for field, values in surprisal.items():
        assert torch.isfinite(values).all(), f"Non-finite surprisal for {field}"
        assert (values >= 0).all(), f"Negative surprisal for {field}"


def test_total_surprisal_is_sum_of_fields(model, sample_context, sample_targets):
    """Total surprisal must equal the sum of the per-field surprisals."""
    model.eval()
    with torch.no_grad():
        surprisal = model.surprisal(sample_context, sample_targets)

    field_sum = sum(surprisal[f] for f in ["eventId", "processName", "userId", "returnValue", "argsNum"])
    torch.testing.assert_close(surprisal["total"], field_sum)


def test_surprisal_varies_with_input(model, vocab_sizes):
    """REGRESSION TRIPWIRE: surprisal must respond when the input changes.

    The next-event analog of the VAE collapse diagnostics — if surprisal
    stops depending on the context, the model has degenerated.
    """
    torch.manual_seed(0)
    batch_size, seq_len = 4, 128

    def make_context(user_id_value, event_value):
        return {
            "processName": torch.full((batch_size, seq_len), 5, dtype=torch.int64),
            "args_ids": torch.full((batch_size, seq_len, 64), 3, dtype=torch.int64),
            "userId": torch.full((batch_size, seq_len), user_id_value, dtype=torch.int64),
            "eventId": torch.full((batch_size, seq_len), event_value, dtype=torch.int64),
            "argsNum": torch.full((batch_size, seq_len), 2, dtype=torch.int64),
            "returnValue": torch.full((batch_size, seq_len), 1, dtype=torch.int64),
            "parentProcessId": torch.full((batch_size, seq_len), 1, dtype=torch.int64),
        }

    def make_targets():
        return {
            "eventId": torch.randint(0, vocab_sizes["event_id_vocab_size"], (batch_size,)),
            "processName": torch.randint(0, vocab_sizes["process_name_vocab_size"], (batch_size,)),
            "userId": torch.randint(0, vocab_sizes["user_id_vocab_size"], (batch_size,)),
            "returnValue": torch.randint(0, vocab_sizes["return_value_vocab_size"], (batch_size,)),
            "argsNum": torch.randint(0, vocab_sizes["args_num_vocab_size"], (batch_size,)),
        }

    targets = make_targets()

    model.eval()
    with torch.no_grad():
        s1 = model.surprisal(make_context(1, 1), targets)["total"]
        s2 = model.surprisal(make_context(2, 3), targets)["total"]

    assert not torch.allclose(s1, s2), (
        "Surprisal identical across different contexts — model ignores its input"
    )


def test_surprisal_varies_with_target(model, sample_context, vocab_sizes):
    """Different targets under the same context should score differently."""
    model.eval()
    t1 = {f: torch.zeros(4, dtype=torch.int64) for f in ["eventId", "processName", "userId", "returnValue", "argsNum"]}
    t2 = {f: torch.ones(4, dtype=torch.int64) for f in t1}
    with torch.no_grad():
        s1 = model.surprisal(sample_context, t1)["total"]
        s2 = model.surprisal(sample_context, t2)["total"]
    assert not torch.allclose(s1, s2)


# ── Serialization ─────────────────────────────────────────────────────────────

def test_init_kwargs_saved(model, vocab_sizes):
    """Model must record its init kwargs for checkpoint round-trips."""
    kwargs = model._init_kwargs
    for key, value in vocab_sizes.items():
        assert kwargs[key] == value, f"{key} not round-tripped"
    assert kwargs["hidden_size"] == 128
    assert kwargs["num_layers"] == 2
    assert kwargs["dropout"] == 0.3


def test_init_kwargs_round_trip(vocab_sizes):
    """Reconstructing from _init_kwargs must yield an equivalent model."""
    model = NextEventLSTM(**vocab_sizes, hidden_size=64, num_layers=1)
    clone = NextEventLSTM(**model._init_kwargs)
    assert clone._init_kwargs == model._init_kwargs
    assert clone.event_dim == model.event_dim
    assert clone.hidden_size == 64
    assert clone.num_layers == 1


# ── Config ────────────────────────────────────────────────────────────────────

def test_hidden_size(model):
    """Default hidden size should be 128."""
    assert model.hidden_size == 128


def test_num_layers(model):
    """Default num_layers should be 2."""
    assert model.num_layers == 2


def test_target_fields(model):
    """The model must predict exactly the five ADR-0004 fields."""
    assert set(model.target_fields) == {"eventId", "processName", "userId", "returnValue", "argsNum"}
