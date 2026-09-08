"""The likelihood-free generative head, and the energy score that trains it.

A softmax head is a probability distribution you can evaluate. This one is
not: it is a *sampler*. It takes the backbone's hidden state and a fresh noise
vector and returns one draw from an implicit distribution over the next latent.
There is no density, so there is no cross-entropy and no perplexity — which is
the price the method pays, and the reason `metrics.py` exists.

What replaces the likelihood is the **energy score**, a strictly proper
scoring rule computable from samples alone:

    ES(P, y) = E‖X − y‖^β − ½ E‖X − X'‖^β,      X, X' ~ P i.i.d.

Minimised over P, it is uniquely minimised at the true distribution. The first
term pulls samples toward the target; the second pushes them apart, and it is
the only thing preventing the head from collapsing onto a single confident
guess. Drop it and you have a regression to the posterior mean.

Two departures from the textbook form, both taken from the CALM reference:

* the target is not a point but the codec's **posterior** `N(mean, std)`, so
  the cross term is estimated with its own draws;
* the returned quantity is `E‖X−X'‖ − 2·E‖X−y‖`, i.e. `−2·ES`, so the loss is
  its negation. The factor of two is a constant and does not move the optimum.

`beta` must lie in (0, 2) for strict propriety. At `beta < 1` the derivative of
`‖x‖^β` is unbounded as `x → 0`, and since the self-distance term evaluates
exactly there whenever two samples coincide, training NaNs. Hence the 1.0
default and the guard in `EnergyHeadConfig`.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import EnergyHeadConfig


def pairwise_power_distance(
    left: torch.Tensor, right: torch.Tensor, beta: float, eps: float = 1e-8
) -> torch.Tensor:
    """‖left − right‖₂^β, broadcast over everything but the last dimension.

    The `eps` inside the norm is load-bearing at `beta < 1`: it keeps the
    gradient finite where two samples coincide.
    """
    difference = left - right
    norm = torch.sqrt(difference.pow(2).sum(dim=-1) + eps)
    return norm.pow(beta)


def energy_score(
    samples: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    beta: float = 1.0,
    num_target_samples: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`E‖X−X'‖^β − 2·E‖X−Y‖^β` for model draws `X` and target draws `Y`.

    Args:
        samples: `(S, ..., latent)` — `S` independent draws from the head.
        mean, log_std: `(..., latent)` — the codec's posterior for the target.
        beta: energy exponent, in (0, 2).
        num_target_samples: draws used for the cross term.

    Returns:
        `(...)` — higher is better. The training loss is its negation.
    """
    num_samples = samples.shape[0]
    if num_samples < 2:
        raise ValueError(
            f"energy_score needs >= 2 samples for the self-distance term, "
            f"got {num_samples}"
        )
    if samples.shape[1:] != mean.shape:
        raise ValueError(
            f"sample shape {tuple(samples.shape[1:])} does not match target "
            f"shape {tuple(mean.shape)}"
        )

    # Self term: mean over ordered pairs. The diagonal contributes exactly
    # zero, so dividing by n(n-1) rather than n^2 debiases it.
    self_distance = pairwise_power_distance(
        samples.unsqueeze(1), samples.unsqueeze(0), beta
    ).sum(dim=(0, 1)) / (num_samples * (num_samples - 1))

    # Cross term against draws from the target posterior.
    std = torch.exp(log_std)
    noise = torch.empty(
        (num_target_samples, *mean.shape), device=mean.device, dtype=mean.dtype
    ).normal_(generator=generator)
    targets = mean + noise * std

    cross_distance = pairwise_power_distance(
        samples.reshape(num_samples, 1, *samples.shape[1:]),
        targets.reshape(1, num_target_samples, *targets.shape[1:]),
        beta,
    ).mean(dim=(0, 1))

    return self_distance - 2.0 * cross_distance


class EnergyMLPBlock(nn.Module):
    """One residual step refining a noise embedding toward a latent.

    The context enters by concatenation rather than addition so the block can
    gate on it: `x` is where the sample lives, `y` is what it must be a sample
    *of*.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.in_norm = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        self.gate_act = nn.SiLU()
        self.down_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.linears(torch.cat((self.in_norm(x), context), dim=-1))
        gate, up = torch.chunk(hidden, 2, dim=-1)
        return x + self.down_proj(self.gate_act(gate) * up)


class EnergyFinalLayer(nn.Module):
    """Projection to latent space, zero-initialised."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.in_norm = nn.LayerNorm(model_channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(self.in_norm(x))


class EnergyHead(nn.Module):
    """Hidden state + noise -> one sampled latent."""

    def __init__(self, config: EnergyHeadConfig):
        super().__init__()
        self.config = config
        self.noise_size = config.noise_size

        self.noise_embed = nn.Linear(config.noise_size, config.hidden_size)
        self.hidden_embed = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_noise = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm_hidden = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.blocks = nn.ModuleList(
            EnergyMLPBlock(config.hidden_size) for _ in range(config.num_mlp_layers)
        )
        self.final_layer = EnergyFinalLayer(config.hidden_size, config.latent_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Start the head as the zero map.

        Every sample is then identical at step 0, the self-distance term is
        exactly zero, and the first gradient comes entirely from the cross
        term — the head learns where the target is before it learns how wide
        it is. Starting from noise instead makes the first steps fight an
        arbitrary spread.
        """
        nn.init.zeros_(self.final_layer.linears[-1].weight)
        nn.init.zeros_(self.final_layer.linears[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one latent per position.

        `noise` is uniform on [-0.5, 0.5], matching the reference: the head is
        free to shape it, so the input distribution only has to be easy to
        sample and bounded.
        """
        if noise is None:
            noise = (
                torch.empty(
                    (*hidden_states.shape[:-1], self.noise_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ).uniform_(generator=generator)
                - 0.5
            )
        noise_embeds = self.norm_noise(self.noise_embed(noise))
        context = self.norm_hidden(self.hidden_embed(hidden_states))
        for block in self.blocks:
            noise_embeds = block(noise_embeds, context)
        return self.final_layer(noise_embeds)

    def sample(
        self,
        hidden_states: torch.Tensor,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """`num_samples` independent draws, stacked on a new leading axis."""
        repeated = hidden_states.unsqueeze(0).expand(
            num_samples, *hidden_states.shape
        )
        return self(repeated, generator=generator)


__all__ = [
    "EnergyHead",
    "EnergyMLPBlock",
    "EnergyFinalLayer",
    "energy_score",
    "pairwise_power_distance",
]
