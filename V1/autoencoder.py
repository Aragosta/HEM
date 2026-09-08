"""The token <-> latent codec: `K` tokens in, one continuous vector out.

This is the piece that makes a continuous autoregressive model possible at
all. The language model downstream never sees a token; it sees a sequence of
latent vectors, each standing for `K` tokens, and it is only as good as this
codec's ability to put `K` tokens into one vector and get them back.

Design, following the CALM reference implementation:

* the encoder is a **VAE** encoder, emitting `(mean, log_std)` rather than a
  point. That is not decoration. The generative head downstream is trained
  against the *posterior* with a proper scoring rule, and a scoring rule needs
  a distribution to score against; a deterministic codec would collapse the
  target and the energy score would degenerate into a regression;
* both halves run in **two stages** with a squeeze (encoder) or expand
  (decoder) in the middle, so the per-token layers see tokens and the
  per-patch layers see patches;
* the decoder's output projection is **tied** to the encoder's embedding, so
  the codec's vocabulary is one matrix, not two;
* reconstruction loss is scaled by `patch_size` during training, because one
  latent is responsible for `K` cross-entropies and the KL must be weighed
  against all of them, not against their mean.

The layers reuse `src.transformer_modules`, so the codec's feed-forward is the
same SwiGLU the baseline Kimi K3 uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.transformer_modules.rms_norm import RMSNorm
from src.transformer_modules.swiglu import SwiGLUFeedForward, SwiGLUMLPConfig

from .config import PatchAutoencoderConfig


@dataclass
class AutoencoderOutput:
    """What one codec forward produces."""

    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor
    logits: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor
    accuracy: torch.Tensor


class _AELayer(nn.Module):
    """Pre-norm residual SwiGLU block. No attention: patches are tiny."""

    def __init__(self, config: PatchAutoencoderConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(
            SwiGLUMLPConfig(
                d_model=config.hidden_size,
                expansion_factor=float(config.ffn_multiplier),
                dropout=config.dropout,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.norm(hidden_states))


class PatchEncoder(nn.Module):
    """`(B, T)` token ids -> `(B, T/K, 2 * latent)` posterior parameters."""

    def __init__(self, config: PatchAutoencoderConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.latent_size = config.latent_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            _AELayer(config) for _ in range(config.num_encoder_layers)
        )
        self.stage_layers = config.num_encoder_layers // 2
        # The squeeze: K per-token vectors become one per-patch vector.
        self.squeeze = nn.Linear(
            config.patch_size * config.hidden_size, config.hidden_size
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.to_latent = nn.Linear(config.hidden_size, config.latent_size * 2)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"expected (batch, seq), got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if seq_len % self.patch_size:
            raise ValueError(
                f"sequence length {seq_len} is not divisible by patch_size "
                f"{self.patch_size}; pad or truncate before encoding"
            )
        num_patches = seq_len // self.patch_size

        hidden = self.embed_tokens(
            input_ids.reshape(batch * num_patches, self.patch_size)
        )
        for index, layer in enumerate(self.layers):
            if index == self.stage_layers:
                hidden = self.squeeze(hidden.reshape(batch * num_patches, 1, -1))
            hidden = layer(hidden)

        latent = self.to_latent(self.norm(hidden))
        latent = latent.reshape(batch, num_patches, self.latent_size * 2)
        mean, log_std = torch.chunk(latent, 2, dim=-1)
        # Keep the posterior inside a range the exponential can represent. An
        # unclamped log_std reaches +-inf on the first bad step and every
        # downstream sample becomes NaN.
        return mean, log_std.clamp(-8.0, 4.0)


class PatchDecoder(nn.Module):
    """`(B, P, latent)` -> `(B, P*K, vocab)` logits."""

    def __init__(self, config: PatchAutoencoderConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        self.from_latent = nn.Linear(config.latent_size, config.hidden_size)
        self.layers = nn.ModuleList(
            _AELayer(config) for _ in range(config.num_decoder_layers)
        )
        self.stage_layers = config.num_decoder_layers // 2
        # The expand: one per-patch vector becomes K per-token vectors.
        self.expand = nn.Linear(
            config.hidden_size, config.patch_size * config.hidden_size
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Set by PatchAutoencoder: tied to the encoder's embedding matrix.
        self.output_weight: torch.Tensor | None = None

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        if latent_states.dim() != 3:
            raise ValueError(
                f"expected (batch, patches, latent), got {tuple(latent_states.shape)}"
            )
        if self.output_weight is None:
            raise RuntimeError(
                "decoder.output_weight is unset; build the decoder through "
                "PatchAutoencoder so it is tied to the encoder embedding"
            )
        batch, num_patches, _ = latent_states.shape

        hidden = self.from_latent(latent_states)
        for index, layer in enumerate(self.layers):
            if index == self.stage_layers:
                hidden = self.expand(hidden)
                hidden = hidden.reshape(batch, num_patches * self.patch_size, -1)
            hidden = layer(hidden)

        return F.linear(self.norm(hidden), self.output_weight)


class PatchAutoencoder(nn.Module):
    """Encoder + decoder + the VAE objective that trains them together."""

    def __init__(self, config: PatchAutoencoderConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.encoder = PatchEncoder(config)
        self.decoder = PatchDecoder(config)
        self.decoder.output_weight = self.encoder.embed_tokens.weight

    @staticmethod
    def reparameterize(
        mean: torch.Tensor, log_std: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        noise = torch.empty_like(mean).normal_(generator=generator)
        return mean + noise * torch.exp(log_std)

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input_ids)

    def decode(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> AutoencoderOutput:
        """Reconstruct `input_ids` through the latent bottleneck.

        `labels` defaults to `input_ids`: the codec is an autoencoder, so the
        target is the input, and a caller only passes labels to reconstruct a
        corrupted input onto a clean target.
        """
        targets = input_ids if labels is None else labels
        mean, log_std = self.encode(input_ids)
        latent = self.reparameterize(mean, log_std)
        if self.training and self.config.dropout > 0.0:
            latent = F.dropout(latent, p=self.config.dropout, training=True)

        logits = self.decode(latent)

        std = torch.exp(log_std)
        kl = 0.5 * (mean.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std)
        kl = kl.clamp(min=self.config.kl_clamp).sum(dim=-1).mean()

        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = targets.reshape(-1)
        reconstruction = F.cross_entropy(flat_logits, flat_targets)
        accuracy = (flat_logits.argmax(-1) == flat_targets).float().mean()

        # One latent answers for `patch_size` tokens, so the reconstruction
        # term is summed over the patch before the KL is weighed against it.
        loss = reconstruction * self.patch_size + kl * self.config.kl_weight
        return AutoencoderOutput(
            loss=loss,
            reconstruction_loss=reconstruction,
            kl_loss=kl,
            logits=logits,
            mean=mean,
            log_std=log_std,
            accuracy=accuracy,
        )

    @torch.no_grad()
    def reconstruction_accuracy(self, input_ids: torch.Tensor) -> float:
        """Token-level accuracy of a full round trip, at the posterior mean.

        This is the ceiling on anything the language model can achieve: no
        amount of latent prediction recovers a token the codec cannot store.
        """
        was_training = self.training
        self.eval()
        mean, _ = self.encode(input_ids)
        logits = self.decode(mean)
        accuracy = (logits.argmax(-1) == input_ids).float().mean().item()
        if was_training:
            self.train()
        return accuracy


__all__ = [
    "AutoencoderOutput",
    "PatchAutoencoder",
    "PatchEncoder",
    "PatchDecoder",
]
