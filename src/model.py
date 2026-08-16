"""
Sentinel — LSTM-VAE Model
Learned embeddings for categorical/text fields + LSTM-based variational
autoencoder for unsupervised anomaly detection on kernel process event sequences.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn


class SentinelVAE(nn.Module):
    """LSTM-based variational autoencoder for anomaly detection.

    Trains on benign event sequences only. Detects anomalies via
    reconstruction error — events the model hasn't seen before
    reconstruct poorly.

    Architecture:
      - Text embeddings: processName (dim=64), args (dim=64)
      - Categorical embeddings: userId (dim=16), mountNamespace (dim=8),
        eventId (dim=16)
      - Numeric features: argsNum, returnValue, parentProcessId (3 scalars)
      - Event vector ≈ 171 dims
      - Encoder LSTM → μ, log σ² (latent dim=32)
      - Reparameterization: z = μ + σ * ε
      - Decoder LSTM → reconstructed event vectors
    """

    def __init__(
        self,
        process_name_vocab_size: int,
        args_vocab_size: int,
        user_id_vocab_size: int,
        mount_ns_vocab_size: int,
        event_id_vocab_size: int,
        process_name_embed_dim: int = 64,
        args_embed_dim: int = 64,
        user_id_embed_dim: int = 16,
        mount_ns_embed_dim: int = 8,
        event_id_embed_dim: int = 16,
        hidden_size: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        process_name_pad_idx: int = 0,
        args_pad_idx: int = 0,
    ):
        super().__init__()

        # ── Embedding layers ─────────────────────────────────────────────────
        self.process_name_embed = nn.Embedding(
            process_name_vocab_size, process_name_embed_dim,
            padding_idx=process_name_pad_idx,
        )
        self.args_embed = nn.Embedding(
            args_vocab_size, args_embed_dim,
            padding_idx=args_pad_idx,
        )
        self.user_id_embed = nn.Embedding(
            user_id_vocab_size, user_id_embed_dim,
            padding_idx=0,
        )
        self.mount_ns_embed = nn.Embedding(
            mount_ns_vocab_size, mount_ns_embed_dim,
            padding_idx=0,
        )
        self.event_id_embed = nn.Embedding(
            event_id_vocab_size, event_id_embed_dim,
            padding_idx=0,
        )

        # ── Event vector dimension ───────────────────────────────────────────
        self.event_dim = (
            process_name_embed_dim
            + args_embed_dim
            + user_id_embed_dim
            + mount_ns_embed_dim
            + event_id_embed_dim
            + 3  # argsNum, returnValue, parentProcessId
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.LSTM(
            input_size=self.event_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(
            input_size=self.event_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(hidden_size, self.event_dim)

        # ── Config ───────────────────────────────────────────────────────────
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._init_kwargs = {
            "process_name_vocab_size": process_name_vocab_size,
            "args_vocab_size": args_vocab_size,
            "user_id_vocab_size": user_id_vocab_size,
            "mount_ns_vocab_size": mount_ns_vocab_size,
            "event_id_vocab_size": event_id_vocab_size,
            "process_name_embed_dim": process_name_embed_dim,
            "args_embed_dim": args_embed_dim,
            "user_id_embed_dim": user_id_embed_dim,
            "mount_ns_embed_dim": mount_ns_embed_dim,
            "event_id_embed_dim": event_id_embed_dim,
            "hidden_size": hidden_size,
            "latent_dim": latent_dim,
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
            "mountNamespace": self.mount_ns_embed.embedding_dim,
            "eventId": self.event_id_embed.embedding_dim,
        }

    def encode(
        self, features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input features into latent distribution parameters.

        Returns:
            (event_vec, mu, logvar) where event_vec is the embedded input
            (B, S, event_dim) and mu/logvar are (B, latent_dim).
        """
        # ── Text embeddings with mean pooling over token dimension ──────────
        pn = features["processName_ids"]
        pn_embedded = self.process_name_embed(pn)
        pn_pooled = pn_embedded.mean(dim=2)

        args = features["args_ids"]
        args_embedded = self.args_embed(args)
        args_pooled = args_embedded.mean(dim=2)

        # ── Categorical embeddings ───────────────────────────────────────────
        uid = self.user_id_embed(features["userId"])
        mns = self.mount_ns_embed(features["mountNamespace"])
        eid = self.event_id_embed(features["eventId"])

        # ── Numeric features ────────────────────────────────────────────────
        anum = features["argsNum"].unsqueeze(-1)
        rval = features["returnValue"].unsqueeze(-1)
        ppid = features["parentProcessId"].unsqueeze(-1)

        # ── Concatenate into event vectors ──────────────────────────────────
        event_vec = torch.cat(
            [pn_pooled, args_pooled, uid, mns, eid, anum, rval, ppid],
            dim=-1,
        )  # (B, S, event_dim)

        # ── Encode ──────────────────────────────────────────────────────────
        _, (hidden, _) = self.encoder(event_vec)  # hidden: (num_layers, B, H)
        last_hidden = hidden[-1]  # (B, H) — top layer's final state

        mu = self.fc_mu(last_hidden)        # (B, latent_dim)
        logvar = self.fc_logvar(last_hidden) # (B, latent_dim)

        return event_vec, mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z = μ + σ * ε using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(
        self, z: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Decode latent vector z into a reconstructed event sequence.

        Args:
            z: Latent vector (B, latent_dim).
            seq_len: Target sequence length.

        Returns:
            Reconstructed event vectors (B, S, event_dim).
        """
        batch_size = z.size(0)
        device = z.device

        # Initialize decoder hidden state from z
        h0 = self.latent_to_hidden(z)  # (B, H)
        # Expand to (num_layers, B, H) by repeating
        h0 = h0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0 = torch.zeros_like(h0)

        # Decoder input: repeat the mean-pooled event vector (use zeros;
        # the decoder relies on hidden state for reconstruction)
        decoder_input = torch.zeros(
            batch_size, seq_len, self.event_dim, device=device
        )

        # Decode
        decoder_out, _ = self.decoder(decoder_input, (h0, c0))
        reconstructed = self.output_projection(decoder_out)  # (B, S, event_dim)

        return reconstructed

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode → reparameterize → decode.

        Args:
            features: Dict of tensors from BethDataset, each of shape
                (batch, seq_len, ...) for token features or
                (batch, seq_len) for scalar features.

        Returns:
            (reconstructed, mu, logvar) where reconstructed is (B, S, event_dim),
            mu and logvar are (B, latent_dim).
        """
        event_vec, mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)
        seq_len = event_vec.size(1)
        reconstructed = self.decode(z, seq_len)
        return reconstructed, mu, logvar

    def reconstruction_error(
        self, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute per-sample reconstruction error (anomaly score).

        Returns:
            MSE per sample, shape (B,). Higher = more anomalous.
        """
        event_vec, mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)
        seq_len = event_vec.size(1)
        reconstructed = self.decode(z, seq_len)

        # Per-sample MSE (mean over sequence and feature dims)
        mse = ((reconstructed - event_vec) ** 2).mean(dim=(1, 2))
        return mse
