"""V1 — Continuous Autoregressive Language Modelling on a Kimi K3 backbone.

Implements the CALM method (Continuous Autoregressive Language Models) over
the vendored Kimi K3 stack in `src/`: a frozen patch autoencoder compresses
`K` tokens into one continuous latent, the K3 backbone runs autoregressively
over those latents, and a likelihood-free energy head predicts the next one.

    from V1 import CalmKimiK3, calm_k3_cpu_tiny_config
    model = CalmKimiK3(calm_k3_cpu_tiny_config())
    loss = model(torch.randint(0, 128, (2, 32))).loss

Nothing here modifies `src/`; the backbone is imported, not forked. See
`V1/README.md` for the method, the training stages and the benchmark.
"""

from .autoencoder import (
    AutoencoderOutput,
    PatchAutoencoder,
    PatchDecoder,
    PatchEncoder,
)
from .config import (
    CalmK3Config,
    EnergyHeadConfig,
    PatchAutoencoderConfig,
    calm_k3_cpu_tiny_config,
)
from .energy_head import EnergyHead, energy_score, pairwise_power_distance
from .generation import generate, generate_baseline
from .metrics import brier_lm, brier_scores, sample_pairs_baseline, sample_pairs_calm
from .model import CalmK3Output, CalmKimiK3, PatchEmbeddingProjector

__all__ = [
    "AutoencoderOutput",
    "CalmK3Config",
    "CalmK3Output",
    "CalmKimiK3",
    "EnergyHead",
    "EnergyHeadConfig",
    "PatchAutoencoder",
    "PatchAutoencoderConfig",
    "PatchDecoder",
    "PatchEmbeddingProjector",
    "PatchEncoder",
    "brier_lm",
    "brier_scores",
    "calm_k3_cpu_tiny_config",
    "energy_score",
    "generate",
    "generate_baseline",
    "pairwise_power_distance",
    "sample_pairs_baseline",
    "sample_pairs_calm",
]
