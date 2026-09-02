"""
Sentinel — Next-Event LSTM Model
Learned embeddings for categorical/text fields + a single forward LSTM
trained benign-only to predict the next event (DeepLog lineage, ADR-0004).
Anomaly score = sum of per-field surprisals: -log P(event i | trailing context).
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NextEventLSTM(nn.Module):
    """Next-event predictor over kernel process event sequences.

    Reads the trailing window of embedded events with a forward LSTM and
    predicts the next event through one cross-entropy head per field:
    eventId, processName, userId, bucketed returnValue, bucketed argsNum.

    Trains on benign sequences only. Detects anomalies via surprisal —
    events the model finds improbable given their past. Posterior collapse
    is structurally impossible: the prediction heads condition on the
    actual input.

    Architecture (per event):
      - processName: exact-match class index → embedding
      - args: token ids → embedding (PAD-masked mean pool)
      - userId, eventId: categorical vocab index → embedding (0 = OOV,
        learnable — UNK-as-input is treated as a novelty signal)
      - argsNum, returnValue: bucket class index → embedding
      - parentProcessId: 3-level index → embedding
      - Event vector ≈ 180 dims → forward LSTM → per-field Linear heads
    """

    target_fields: Tuple[str, ...] = ("eventId", "processName", "userId", "returnValue", "argsNum")

    def __init__(
        self,
        process_name_vocab_size: int,
        args_vocab_size: int,
        user_id_vocab_size: int,
        event_id_vocab_size: int,
        args_num_vocab_size: int = 16,
        return_value_vocab_size: int = 13,
        parent_pid_vocab_size: int = 3,
        process_name_embed_dim: int = 64,
        args_embed_dim: int = 64,
        user_id_embed_dim: int = 16,
        event_id_embed_dim: int = 16,
        args_num_embed_dim: int = 8,
        return_value_embed_dim: int = 8,
        parent_pid_embed_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        args_pad_idx: int = 0,
    ):
        super().__init__()

        # ── Embedding layers ─────────────────────────────────────────────────
        # Single-index categorical fields: index 0 is OOV (a learnable novelty
        # signal), so NO padding_idx — freezing row 0 would erase the signal.
        self.process_name_embed = nn.Embedding(
            process_name_vocab_size, process_name_embed_dim
        )
        self.user_id_embed = nn.Embedding(user_id_vocab_size, user_id_embed_dim)
        self.event_id_embed = nn.Embedding(event_id_vocab_size, event_id_embed_dim)
        self.args_num_embed = nn.Embedding(args_num_vocab_size, args_num_embed_dim)
        self.return_value_embed = nn.Embedding(return_value_vocab_size, return_value_embed_dim)
        self.parent_pid_embed = nn.Embedding(parent_pid_vocab_size, parent_pid_embed_dim)

        # args token ids only: index 0 is genuine padding → padding_idx + masked pool
        self.args_embed = nn.Embedding(
            args_vocab_size, args_embed_dim, padding_idx=args_pad_idx
        )
        self.args_pad_idx = args_pad_idx

        # ── Event vector dimension ───────────────────────────────────────────
        self.event_dim = (
            process_name_embed_dim
            + args_embed_dim
            + user_id_embed_dim
            + event_id_embed_dim
            + args_num_embed_dim
            + return_value_embed_dim
            + parent_pid_embed_dim
        )

        # ── Sequence encoder ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.event_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Per-field prediction heads ───────────────────────────────────────
        head_sizes = {
            "processName": process_name_vocab_size,
            "userId": user_id_vocab_size,
            "eventId": event_id_vocab_size,
            "returnValue": return_value_vocab_size,
            "argsNum": args_num_vocab_size,
        }
        self.heads = nn.ModuleDict({
            field: nn.Linear(hidden_size, size) for field, size in head_sizes.items()
        })

        # ── Config ───────────────────────────────────────────────────────────
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._init_kwargs = {
            "process_name_vocab_size": process_name_vocab_size,
            "args_vocab_size": args_vocab_size,
            "user_id_vocab_size": user_id_vocab_size,
            "event_id_vocab_size": event_id_vocab_size,
            "args_num_vocab_size": args_num_vocab_size,
            "return_value_vocab_size": return_value_vocab_size,
            "parent_pid_vocab_size": parent_pid_vocab_size,
            "process_name_embed_dim": process_name_embed_dim,
            "args_embed_dim": args_embed_dim,
            "user_id_embed_dim": user_id_embed_dim,
            "event_id_embed_dim": event_id_embed_dim,
            "args_num_embed_dim": args_num_embed_dim,
            "return_value_embed_dim": return_value_embed_dim,
            "parent_pid_embed_dim": parent_pid_embed_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    @property
    def embedding_dims(self) -> Dict[str, int]:
        """Return the configured embedding dimension for each feature."""
        return {
            "processName": self.process_name_embed.embedding_dim,
            "args": self.args_embed.embedding_dim,
            "userId": self.user_id_embed.embedding_dim,
            "eventId": self.event_id_embed.embedding_dim,
            "argsNum": self.args_num_embed.embedding_dim,
            "returnValue": self.return_value_embed.embedding_dim,
            "parentProcessId": self.parent_pid_embed.embedding_dim,
        }

    def embed_context(
        self, context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Embed each context event into a dense vector.

        Args:
            context: Dict of tensors — token features (B, S, L), single-index
                features (B, S).

        Returns:
            Event vectors (B, S, event_dim).
        """
        # args: token ids (B, S, L) → embed (B, S, L, E) → PAD-masked mean pool
        args_ids = context["args_ids"]
        args_embedded = self.args_embed(args_ids)
        pad_mask = (args_ids != self.args_pad_idx).unsqueeze(-1)
        args_summed = (args_embedded * pad_mask).sum(dim=2)
        args_pooled = args_summed / pad_mask.sum(dim=2).clamp(min=1)

        pn = self.process_name_embed(context["processName"])
        uid = self.user_id_embed(context["userId"])
        eid = self.event_id_embed(context["eventId"])
        anum = self.args_num_embed(context["argsNum"])
        rval = self.return_value_embed(context["returnValue"])
        ppid = self.parent_pid_embed(context["parentProcessId"])

        return torch.cat([pn, args_pooled, uid, eid, anum, rval, ppid], dim=-1)

    def forward(
        self, context: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict the next event's fields from the trailing context.

        Args:
            context: Dict of tensors from TrailingWindowDataset — token
                features (B, S, L), single-index features (B, S).

        Returns:
            Dict of per-field logits, each (B, vocab_k), keyed by field.
        """
        event_vec = self.embed_context(context)  # (B, S, event_dim)

        _, (hidden, _) = self.lstm(event_vec)  # hidden: (num_layers, B, H)
        last_hidden = hidden[-1]  # (B, H) — top layer's final state

        return {field: head(last_hidden) for field, head in self.heads.items()}

    def surprisal(
        self,
        context: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-field surprisal of the targets given the context.

        Args:
            context: Dict of context tensors (see forward).
            targets: Dict of scalar target class indices, each (B,).

        Returns:
            Dict with one (B,) surprisal tensor per target field plus
            'total' — their sum, the joint surprisal under the factorized
            heads. Higher = the model finds the event more improbable.
        """
        logits = self.forward(context)
        out = {
            field: F.cross_entropy(
                logits[field], targets[field], reduction="none"
            )
            for field in self.target_fields
        }
        out["total"] = sum(out.values())
        return out
