"""CALM-K3: a Kimi K3 backbone made continuous and autoregressive.

The baseline Kimi K3 reads `T` tokens, runs `T` positions through the
KDA/MLA + Stable LatentMoE + AttnRes stack, and predicts a distribution over
the vocabulary at each one. This model does the same thing over *patches*:

    tokens ──embed──▶ group K ──▶ project ──▶ K3 backbone ──▶ energy head
                                                  (T/K positions)      │
    tokens ◀──decode── frozen codec ◀──────── next latent ◀────────────┘

Three consequences, and they are the whole trade:

1. **The backbone runs on `T/K` positions.** Every quadratic term in the Gated
   MLA layers falls by `K²`, every recurrent step in the KDA layers by `K`.
   This is where the compute goes.
2. **The output is continuous.** There is no softmax over the vocabulary and
   therefore no likelihood, no cross-entropy and no perplexity. Training uses
   the energy score (`energy_head.py`); evaluation uses BrierLM
   (`metrics.py`), which needs only samples and so can also score the
   baseline.
3. **The codec is a ceiling.** Whatever the backbone predicts is decoded by a
   frozen autoencoder, so no token the codec cannot reconstruct is reachable.
   `PatchAutoencoder.reconstruction_accuracy` measures that ceiling, and it
   should be checked before any language-model number is believed.

The backbone is `src.kimi_block.KimiBlock`, unmodified and unwrapped — the
same class the baseline builds — so a difference between the two models is a
difference in what feeds it and what reads it, never in the stack itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.hybrid_backbone.cache import HybridBackboneCache
from src.kimi_block import KimiBlock

from .autoencoder import PatchAutoencoder
from .config import CalmK3Config
from .energy_head import EnergyHead, energy_score


@dataclass
class CalmK3Output:
    """One CALM-K3 forward."""

    loss: torch.Tensor | None
    hidden_states: torch.Tensor
    samples: torch.Tensor | None
    target_mean: torch.Tensor | None
    target_log_std: torch.Tensor | None
    cache: HybridBackboneCache | None = None


class PatchEmbeddingProjector(nn.Module):
    """`K` token embeddings -> one backbone input vector."""

    def __init__(self, d_model: int, patch_size: int, eps: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.project = nn.Sequential(
            nn.Linear(patch_size * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model, eps=eps),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = token_embeddings.shape
        if seq_len % self.patch_size:
            raise ValueError(
                f"sequence length {seq_len} is not divisible by patch_size "
                f"{self.patch_size}"
            )
        grouped = token_embeddings.reshape(
            batch, seq_len // self.patch_size, self.patch_size * d_model
        )
        return self.project(grouped)


class CalmKimiK3(nn.Module):
    """Continuous autoregressive language model on a Kimi K3 backbone."""

    def __init__(self, config: CalmK3Config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_projector = PatchEmbeddingProjector(
            config.d_model, config.patch_size, eps=config.backbone.rms_norm_eps
        )
        self.backbone = KimiBlock(config.backbone)
        self.energy_head = EnergyHead(config.energy_head)

        # The codec is a fixed part of the environment, not of the model being
        # trained: it defines what a latent *means*, and a moving target would
        # make the energy score chase its own tail.
        self.autoencoder = PatchAutoencoder(config.autoencoder)
        self.freeze_autoencoder()

        self.apply(self._init_weights)
        self.energy_head.initialize_weights()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # The backbone and codec initialise themselves; only the pieces
            # this class owns are touched here.
            if module is self.embed_tokens or self._owned_by_projector(module):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _owned_by_projector(self, module: nn.Module) -> bool:
        return any(module is child for child in self.patch_projector.modules())

    # ---------------------------------------------------------------- codec

    def freeze_autoencoder(self) -> None:
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)
        self.autoencoder.eval()

    def load_autoencoder(self, autoencoder: PatchAutoencoder) -> None:
        """Adopt a separately trained codec, then freeze it."""
        if autoencoder.config.latent_size != self.config.autoencoder.latent_size:
            raise ValueError("codec latent_size does not match this model's")
        if autoencoder.config.patch_size != self.patch_size:
            raise ValueError("codec patch_size does not match this model's")
        self.autoencoder = autoencoder
        self.freeze_autoencoder()

    def train(self, mode: bool = True) -> "CalmKimiK3":
        super().train(mode)
        # A frozen codec stays in eval whatever the rest of the model does,
        # or its dropout would inject noise into the training target.
        self.autoencoder.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------- forward

    def encode_patches(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> one backbone input vector per patch."""
        return self.patch_projector(self.embed_tokens(input_ids))

    def run_backbone(
        self,
        patch_embeddings: torch.Tensor,
        cache: HybridBackboneCache | None = None,
        use_cache: bool = False,
        mode: str = "full",
    ):
        return self.backbone(
            patch_embeddings, cache=cache, use_cache=use_cache, mode=mode
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        compute_loss: bool = True,
        generator: torch.Generator | None = None,
    ) -> CalmK3Output:
        """Predict every patch from the ones before it.

        Position `p` of the backbone sees patches `0..p` and is trained to
        produce patch `p+1`, so a `P`-patch sequence yields `P-1` predictions
        — the same off-by-one as next-token prediction, one patch coarser.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"expected (batch, seq), got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if seq_len % self.patch_size:
            raise ValueError(
                f"sequence length {seq_len} is not divisible by patch_size "
                f"{self.patch_size}"
            )
        num_patches = seq_len // self.patch_size
        if num_patches < 2:
            raise ValueError(
                f"need at least 2 patches to form one prediction, got "
                f"{num_patches} (seq_len {seq_len}, patch_size {self.patch_size})"
            )

        patch_embeddings = self.encode_patches(input_ids)[:, :-1]
        hidden_states = self.run_backbone(patch_embeddings).last_hidden_state

        if not compute_loss:
            return CalmK3Output(
                loss=None,
                hidden_states=hidden_states,
                samples=None,
                target_mean=None,
                target_log_std=None,
            )

        with torch.no_grad():
            mean, log_std = self.autoencoder.encode(input_ids)
        target_mean, target_log_std = mean[:, 1:], log_std[:, 1:]

        samples = self.energy_head.sample(
            hidden_states, self.config.energy_head.num_samples, generator=generator
        )
        score = energy_score(
            samples,
            target_mean,
            target_log_std,
            beta=self.config.energy_head.beta,
            num_target_samples=self.config.energy_head.num_target_samples,
            generator=generator,
        )
        return CalmK3Output(
            loss=-score.mean(),
            hidden_states=hidden_states,
            samples=samples,
            target_mean=target_mean,
            target_log_std=target_log_std,
        )

    # ------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample_next_latent(
        self,
        input_ids: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One latent for the patch that follows `input_ids`."""
        patch_embeddings = self.encode_patches(input_ids)
        hidden = self.run_backbone(patch_embeddings).last_hidden_state[:, -1]
        return self.energy_head(hidden, generator=generator)

    @torch.no_grad()
    def predict_tokens(
        self,
        input_ids: torch.Tensor,
        generator: torch.Generator | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Teacher-forced next-patch token predictions, `(B, P-1, K)`.

        Every patch is predicted from the true prefix, which is the
        conditioning BrierLM assumes on both models.
        """
        patch_embeddings = self.encode_patches(input_ids)[:, :-1]
        hidden = self.run_backbone(patch_embeddings).last_hidden_state
        latents = self.energy_head(hidden, generator=generator)
        logits = self.autoencoder.decode(latents)
        batch, num_predicted, _ = latents.shape
        if temperature <= 0.0:
            tokens = logits.argmax(-1)
        else:
            probabilities = torch.softmax(logits.float() / temperature, dim=-1)
            tokens = torch.multinomial(
                probabilities.reshape(-1, probabilities.size(-1)),
                num_samples=1,
                generator=generator,
            ).reshape(probabilities.shape[:-1])
        return tokens.reshape(batch, num_predicted, self.patch_size)


__all__ = ["CalmKimiK3", "CalmK3Output", "PatchEmbeddingProjector"]
