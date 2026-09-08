"""Configuration for the continuous-autoregressive Kimi K3 (CALM-K3).

Three configs, one per trainable object:

``PatchAutoencoderConfig``
    The token <-> latent codec. Trained first, then frozen. It decides `K`
    (tokens per patch) and the latent width, which together set the
    compression ratio the whole method rests on.

``EnergyHeadConfig``
    The likelihood-free generative head. Its `num_samples` and `beta` are the
    estimator's knobs, not architecture: `beta` is the exponent in the energy
    distance and must stay in (0, 2); at `beta < 1` the self-distance term has
    an unbounded derivative at zero and training NaNs, which is why the
    default is exactly 1.0.

``CalmK3Config``
    Ties the codec and the head to a Kimi K3 backbone (`KimiBlockConfig`), so
    the sequence model underneath is the same KDA/MLA + Stable LatentMoE +
    AttnRes stack the baseline uses, only running on `T/K` positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.kimi_block import KimiBlockConfig
# `_build_text_configs` is the same private helper the baseline's own CPU
# factories use. Importing it keeps the two backbones structurally identical
# by construction; rebuilding a KimiBlockConfig by hand here would let the
# comparison drift the moment upstream changes a default.
from src.kimi_k3.config import _build_text_configs, kimi_k3_cpu_tiny_config


@dataclass
class PatchAutoencoderConfig:
    """Token <-> latent codec over fixed patches of `patch_size` tokens."""

    vocab_size: int
    patch_size: int = 4
    hidden_size: int = 128
    latent_size: int = 32
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    ffn_multiplier: int = 4
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    kl_weight: float = 1.0
    # Per-dimension floor on the KL. Without it the codec drives log_std to
    # -inf, the posterior becomes a point mass, and the energy target the head
    # is asked to match loses the spread that makes the score proper.
    kl_clamp: float = 0.0

    def __post_init__(self) -> None:
        if self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.latent_size < 1 or self.hidden_size < 1:
            raise ValueError("latent_size and hidden_size must be >= 1")
        if self.num_encoder_layers % 2 or self.num_decoder_layers % 2:
            raise ValueError(
                "encoder/decoder layer counts must be even: each runs in two "
                "stages, before and after the squeeze/expand"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def compression(self) -> float:
        """Tokens carried per latent vector. The whole point of the method."""
        return float(self.patch_size)


@dataclass
class EnergyHeadConfig:
    """Likelihood-free generative head: noise + context -> one latent."""

    hidden_size: int
    latent_size: int
    noise_size: int = 32
    num_mlp_layers: int = 3
    # Samples drawn per position to estimate the energy score. The estimator
    # needs >= 2 to have a self-distance term at all.
    num_samples: int = 8
    # Draws from the target posterior for the cross term.
    num_target_samples: int = 100
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError(
                "num_samples must be >= 2; the pairwise self-distance term "
                "divides by n(n-1)"
            )
        if self.num_target_samples < 1:
            raise ValueError("num_target_samples must be >= 1")
        if not 0.0 < self.beta < 2.0:
            raise ValueError(
                "beta must lie in (0, 2) for the energy score to be strictly "
                "proper; below 1.0 the self-distance derivative is unbounded "
                "at zero and training diverges"
            )


@dataclass
class CalmK3Config:
    """A Kimi K3 backbone driven autoregressively over continuous latents."""

    vocab_size: int
    d_model: int
    backbone: KimiBlockConfig
    autoencoder: PatchAutoencoderConfig
    energy_head: EnergyHeadConfig
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.backbone.d_model != self.d_model:
            raise ValueError("backbone.d_model must match d_model")
        if self.autoencoder.vocab_size != self.vocab_size:
            raise ValueError("autoencoder.vocab_size must match vocab_size")
        if self.energy_head.latent_size != self.autoencoder.latent_size:
            raise ValueError(
                "energy_head.latent_size must match the codec's latent_size: "
                "the head predicts exactly what the decoder consumes"
            )
        if self.energy_head.hidden_size != self.d_model:
            raise ValueError("energy_head.hidden_size must match d_model")

    @property
    def patch_size(self) -> int:
        return self.autoencoder.patch_size


def calm_k3_cpu_tiny_config(
    patch_size: int = 4,
    latent_size: int = 16,
    vocab_size: int | None = None,
) -> CalmK3Config:
    """A CPU-sized CALM-K3 sharing the baseline's tiny backbone.

    The backbone is taken from ``kimi_k3_cpu_tiny_config`` unchanged, so a
    CALM-K3 and a baseline Kimi K3 built from these two factories differ only
    in what feeds the stack and what reads it out.
    """
    base = kimi_k3_cpu_tiny_config()
    vocab = vocab_size if vocab_size is not None else base.vocab_size
    return CalmK3Config(
        vocab_size=vocab,
        d_model=base.d_model,
        backbone=base.backbone,
        autoencoder=PatchAutoencoderConfig(
            vocab_size=vocab,
            patch_size=patch_size,
            hidden_size=base.d_model,
            latent_size=latent_size,
            num_encoder_layers=2,
            num_decoder_layers=2,
        ),
        energy_head=EnergyHeadConfig(
            hidden_size=base.d_model,
            latent_size=latent_size,
            noise_size=latent_size,
            num_mlp_layers=2,
            num_samples=4,
            num_target_samples=16,
        ),
        pad_token_id=base.pad_token_id,
        eos_token_id=base.eos_token_id,
    )


def calm_k3_cpu_small_config(
    patch_size: int = 4,
    latent_size: int = 32,
    vocab_size: int = 128,
    d_model: int = 128,
) -> CalmK3Config:
    """A backbone big enough that the codec is not the only thing being tested.

    The tiny factory's `d_model = 16` cannot carry a 4-token patch: its codec
    tops out around 25% reconstruction, and since the codec is a hard ceiling
    every language-model number underneath it measures the bottleneck instead
    of the model. This one widens both, and is what `benchmark.py` uses.

    The returned backbone config is exactly what `KimiK3` would be given at the
    same width, so the baseline in the benchmark is a fair opposite number.
    """
    backbone, _kda, _mla, _moe = _build_text_configs(
        d_model,
        num_heads=4,
        key_head_dim=32,
        q_head_dim=32,
        latent_dim=64,
        shared_hidden=256,
        routed_hidden=128,
        num_routed=4,
        top_k=2,
        num_shared=1,
        groups=1,
        attnres_sublayers=4,
    )
    return CalmK3Config(
        vocab_size=vocab_size,
        d_model=d_model,
        backbone=backbone,
        autoencoder=PatchAutoencoderConfig(
            vocab_size=vocab_size,
            patch_size=patch_size,
            hidden_size=128,
            latent_size=latent_size,
            num_encoder_layers=4,
            num_decoder_layers=4,
        ),
        energy_head=EnergyHeadConfig(
            hidden_size=d_model,
            latent_size=latent_size,
            noise_size=latent_size,
            num_mlp_layers=3,
            num_samples=8,
            num_target_samples=32,
        ),
        pad_token_id=0,
        eos_token_id=2,
    )


__all__ = [
    "PatchAutoencoderConfig",
    "EnergyHeadConfig",
    "CalmK3Config",
    "calm_k3_cpu_tiny_config",
    "calm_k3_cpu_small_config",
    "replace",
    "field",
]
