"""HELM-CALM: the optimized HELM backbone with CALM's continuous head.

Assembles, into one model, the pieces the staged experiments validated:

* the **optimized** HELM-MiCE backbone from :mod:`helm.modules.helm_mice`,
  imported unmodified — flash attention, sorted MoE dispatch, fused residual and
  fused experts all still on;
* a **Lorentz patch embedding** collapsing K token vectors into one while staying
  on the manifold;
* CALM's **energy head**, predicting a continuous latent instead of a
  distribution over 128256 tokens;
* a **frozen autoencoder** supplying the target latents.

Nothing in ``helm/`` is touched. The vocabulary head is removed from the backbone
instance at construction, which is a per-instance mutation, not a change to the
class.

Settled by the experiments in ``experiments/`` and baked in here as defaults:
``lr ~1e-3`` (see :meth:`HelmCALM.parameter_groups`), ``beta=1.0`` (below 1 the
energy score's self-distance term has an unbounded derivative and produces NaN
gradients), and a Riemannian update for ``ManifoldParameter``s (see
:meth:`HelmCALM.retract_manifold_parameters`).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from helm.hypercore.manifolds import Lorentz
from helm.hypercore.nn.linear.lorentz_linear import LorentzLinear
from helm.modules.helm_mice import HelmMiCE

__all__ = ["LorentzPatchEmbedding", "CalmEnergyHead", "PatchAutoencoder",
           "HelmCALM", "energy_score"]


# --------------------------------------------------------------------- pieces

class LorentzPatchEmbedding(nn.Module):
    """Collapse K Lorentz token vectors into one, staying on the manifold.

    CALM concatenates K Euclidean token embeddings and projects them. HELM's
    embeddings are points on a hyperboloid, and concatenating those lands on no
    manifold at all. Here the K *space-like* parts are concatenated and the time
    coordinate recomputed — a valid point in a wider Minkowski space — then a
    ``LorentzLinear`` maps it back onto the model's manifold.
    """

    def __init__(self, manifold: Lorentz, dim: int, patch_size: int):
        super().__init__()
        self.manifold = manifold
        self.patch_size = patch_size
        self.dim = dim
        self.proj = LorentzLinear(manifold, patch_size * (dim - 1) + 1, dim - 1)

    def forward(self, tokens_on_manifold: torch.Tensor) -> torch.Tensor:
        """``(B, S, dim)`` -> ``(B, S // patch_size, dim)``."""
        space = tokens_on_manifold[..., 1:]
        batch, seqlen, width = space.shape
        if seqlen % self.patch_size:
            raise ValueError(
                f"sequence length {seqlen} is not a multiple of patch size "
                f"{self.patch_size}")
        space = space.reshape(batch, seqlen // self.patch_size,
                              self.patch_size * width)
        time = (space.square().sum(-1, keepdim=True)
                + self.manifold.c).clamp_min(1e-8).sqrt()
        return self.proj(torch.cat([time, space], dim=-1))


class CalmEnergyHead(nn.Module):
    """CALM's ``MLPGenerator``: a hidden state plus noise becomes a latent.

    Transcribed from ``upstream/models/modeling_energy.py``. The final projection
    is zero-initialised, as upstream does, so the head starts neutral.
    """

    class Block(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.in_ln = nn.LayerNorm(channels, eps=1e-6)
            self.linears = nn.Sequential(
                nn.Linear(2 * channels, channels), nn.SiLU(),
                nn.Linear(channels, channels), nn.SiLU(),
                nn.Linear(channels, 2 * channels))
            self.gate_act = nn.SiLU()
            self.down_proj = nn.Linear(channels, channels)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
            gate, up = torch.chunk(h, 2, dim=-1)
            return x + self.down_proj(self.gate_act(gate) * up)

    def __init__(self, hidden_size: int, latent_size: int, noise_size: int = 64,
                 num_mlp_layers: int = 4):
        super().__init__()
        self.noise_size = noise_size
        self.noise_embd = nn.Linear(noise_size, hidden_size)
        self.hidden_embd = nn.Linear(hidden_size, hidden_size)
        self.norm_noise = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_hidden = nn.LayerNorm(hidden_size, eps=1e-6)
        self.blocks = nn.ModuleList(
            [self.Block(hidden_size) for _ in range(num_mlp_layers)])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, latent_size))
        nn.init.zeros_(self.final_layer[-1].weight)
        nn.init.zeros_(self.final_layer[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(..., hidden)`` -> ``(..., latent)``, one stochastic draw."""
        noise = torch.rand((*hidden_states.shape[:-1], self.noise_size),
                           dtype=hidden_states.dtype,
                           device=hidden_states.device) - 0.5
        h = self.norm_noise(self.noise_embd(noise))
        y = self.norm_hidden(self.hidden_embd(hidden_states))
        for block in self.blocks:
            h = block(h, y)
        return self.final_layer(h)


def energy_score(samples: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor,
                 beta: float = 1.0, n_target: int = 100) -> torch.Tensor:
    """CALM's energy score. Verified bit-identical to upstream, gradients included.

    Strictly proper for ``beta`` in (0, 2). **Do not set beta below 1**: the
    pairwise term includes the self-distances ``||x_i - x_i|| = 0``, where
    ``d/dx ||x||^beta`` is unbounded for ``beta < 1``, producing NaN gradients —
    in CALM's implementation as much as this one.

    Args:
        samples: ``(n_samples, tokens, latent)`` draws from the head.
        mean, log_std: ``(tokens, latent)`` target posterior from the autoencoder.
        beta: exponent on the distance.
        n_target: Monte-Carlo draws from the target distribution.

    Returns:
        ``(tokens,)`` score; the training loss is its negation.
    """
    if not 0 < beta < 2:
        raise ValueError(f"beta must lie in (0, 2), got {beta}")

    def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(a - b, ord=2, dim=-1).pow(beta)

    n_samples = samples.shape[0]
    pairwise = distance(samples.unsqueeze(1), samples.unsqueeze(0))
    distance_x = pairwise.sum(dim=(0, 1)) / (n_samples * (n_samples - 1))

    targets = mean + torch.randn((n_target, *mean.shape), device=mean.device,
                                 dtype=mean.dtype) * log_std.exp()
    cross = distance(samples.reshape(n_samples, 1, *samples.shape[1:]),
                     targets.reshape(1, n_target, *targets.shape[1:]))
    return distance_x - cross.mean(dim=(0, 1)) * 2


class PatchAutoencoder(nn.Module):
    """Reference K-token autoencoder, the shape of CALM's.

    Structure follows ``upstream/models/modeling_autoencoder.py``: embed K
    tokens, MLP blocks, squeeze to one vector, project to a Gaussian posterior;
    then expand back and decode through a tied head. Attention-free, as upstream.

    For a real run, load CALM's released 75.8M checkpoint instead — it is
    tokenizer-compatible with HELM (both 128256-entry Llama-3). This class exists
    so the model is runnable and testable without that download.
    """

    class Block(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.norm = nn.RMSNorm(hidden, eps=1e-5)
            self.gate = nn.Linear(hidden, 4 * hidden, bias=False)
            self.up = nn.Linear(hidden, 4 * hidden, bias=False)
            self.down = nn.Linear(4 * hidden, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def __init__(self, vocab_size: int, hidden: int = 256, latent_size: int = 128,
                 patch_size: int = 4, layers: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.enc_a = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.squeeze = nn.Linear(patch_size * hidden, hidden)
        self.enc_b = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.enc_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.to_latent = nn.Linear(hidden, latent_size * 2)
        self.from_latent = nn.Linear(latent_size, hidden)
        self.dec_a = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.expand = nn.Linear(hidden, patch_size * hidden)
        self.dec_b = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.dec_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def encode(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(N, K)`` ids -> ``(mean, log_std)``, each ``(N, latent)``."""
        h = self.embed(ids)
        for block in self.enc_a:
            h = block(h)
        h = self.squeeze(h.flatten(-2))
        for block in self.enc_b:
            h = block(h)
        return self.to_latent(self.enc_norm(h)).chunk(2, dim=-1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``(..., latent)`` -> ``(..., K, vocab)``. Shape-agnostic in front."""
        h = self.from_latent(latent)
        for block in self.dec_a:
            h = block(h)
        h = self.expand(h).view(*latent.shape[:-1], self.patch_size, -1)
        for block in self.dec_b:
            h = block(h)
        return self.head(self.dec_norm(h))

    def elbo(self, ids: torch.Tensor, kl_weight: float = 1e-3):
        """Reconstruction + KL, for pretraining. Returns ``(loss, recon_ce)``."""
        mean, log_std = self.encode(ids)
        latent = mean + torch.randn_like(mean) * log_std.exp()
        logits = self.decode(latent)
        recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
        kl = (0.5 * (mean.pow(2) + (2 * log_std).exp() - 1) - log_std).sum(-1).mean()
        return recon * self.patch_size + kl_weight * kl, recon

    def freeze(self) -> "PatchAutoencoder":
        """Put into the state CALM keeps it in: no gradients, eval mode."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self.eval()


# ---------------------------------------------------------------------- model

class HelmCALM(nn.Module):
    """Optimized HELM-MiCE predicting continuous latents instead of tokens.

    Args:
        args: HELM model config (``config/args.py`` or ``helm.eval.presets``).
        autoencoder: a frozen autoencoder exposing ``encode``/``decode`` and
            ``latent_size``/``patch_size``.
        manifolds: input/hidden/output manifolds; defaults to three unit Lorentz.
        num_samples: draws per position for the energy score. CALM uses 8.
        beta: energy-score exponent; see :func:`energy_score`.
        backbone_kwargs: forwarded to :class:`~helm.modules.helm_mice.HelmMiCE`,
            so every optimization stays selectable.

    The three heads of the model are deliberately separate methods —
    :meth:`loss`, :meth:`sample_tokens`, :meth:`hidden_states` — rather than one
    ``forward`` that returns different things depending on ``self.training``.
    That pattern is what makes upstream's ``LorentzMoE`` unusable in eval mode.
    """

    def __init__(self, args, autoencoder, manifolds: Optional[Tuple] = None,
                 num_samples: int = 8, beta: float = 1.0, **backbone_kwargs):
        super().__init__()
        if manifolds is None:
            manifolds = (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
        self.patch_size = autoencoder.patch_size
        self.num_samples = num_samples
        self.beta = beta
        self.dim = args.dim
        self.vocab_size = args.vocab_size

        self.backbone = HelmMiCE(args, *manifolds, **backbone_kwargs)
        # The vocabulary projection is what CALM replaces. Removing it from this
        # instance leaves helm.modules.helm_mice untouched.
        del self.backbone.head

        self.patch_embed = LorentzPatchEmbedding(manifolds[1], args.dim,
                                                 self.patch_size)
        self.head = CalmEnergyHead(args.dim, autoencoder.latent_size)
        self.autoencoder = autoencoder.freeze()

    # ---------------------------------------------------------------- plumbing

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, S)`` token ids -> ``(B, S // K, dim)`` per-patch hidden states."""
        backbone = self.backbone
        embedded = backbone.embed(tokens)
        h = self.patch_embed(embedded)
        freqs_cis = backbone.freqs_cis[:h.size(1)]
        for layer in backbone.layers:
            out = layer(h, 0, freqs_cis, None, True, None)
            h = out[0] if isinstance(out, tuple) else out
        return backbone.norm(backbone.final_proj(h, return_space=True),
                             space_only=True)

    def _aligned(self, tokens: torch.Tensor):
        """Split into input patches and the next-patch targets they predict."""
        n_patches = tokens.size(1) // self.patch_size
        if n_patches < 2:
            raise ValueError("need at least two patches to form a prediction")
        inputs = tokens[:, :(n_patches - 1) * self.patch_size]
        targets = tokens[:, self.patch_size:n_patches * self.patch_size]
        return inputs, targets.reshape(-1, self.patch_size)

    # ------------------------------------------------------------------ heads

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Energy loss for next-patch prediction. ``(B, S)`` -> scalar."""
        inputs, targets = self._aligned(tokens)
        with torch.no_grad():
            mean, log_std = self.autoencoder.encode(targets)
        hidden = self.hidden_states(inputs).reshape(-1, self.dim)
        samples = self.head(hidden.unsqueeze(0).expand(self.num_samples, -1, -1))
        return -energy_score(samples, mean, log_std, beta=self.beta).mean()

    @torch.no_grad()
    def sample_tokens(self, tokens: torch.Tensor, n_samples: Optional[int] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Independent token samples for the next patch of every position.

        Returns ``(samples, targets)`` with samples of shape
        ``(n_samples, positions, K)`` — the form :mod:`brierlm` consumes.
        """
        n_samples = n_samples or self.num_samples
        inputs, targets = self._aligned(tokens)
        hidden = self.hidden_states(inputs).reshape(-1, self.dim)
        latents = self.head(hidden.unsqueeze(0).expand(n_samples, -1, -1))
        return self.autoencoder.decode(latents).argmax(-1), targets

    @torch.no_grad()
    def predict_tokens(self, tokens: torch.Tensor, n_samples: int = 32
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modal prediction, by majority vote over a sample pool."""
        samples, targets = self.sample_tokens(tokens, n_samples)
        return torch.mode(samples, dim=0).values, targets

    # --------------------------------------------------------------- training

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Parameters split by how they must be optimized.

        ``manifold`` holds the ``ManifoldParameter``s, which live on the
        hyperboloid and need a Riemannian update (or the retraction below).
        Optimizing them with plain AdamW drove the constraint violation to 3.92
        in the Stage 1 experiments — see ``RESULTS.md``.
        """
        manifold_ids = {id(p) for p in self.manifold_parameters()}
        euclidean = [p for p in self.parameters()
                     if p.requires_grad and id(p) not in manifold_ids]
        return {"euclidean": euclidean,
                "manifold": [p for p in self.manifold_parameters()
                             if p.requires_grad]}

    def manifold_parameters(self) -> Iterator[nn.Parameter]:
        """The parameters constrained to lie on the hyperboloid."""
        yield self.backbone.embed.embedding

    @torch.no_grad()
    def retract_manifold_parameters(self) -> None:
        """Project manifold parameters back onto the hyperboloid.

        Call after each optimizer step when using a Euclidean optimizer for them.
        A Riemannian optimizer maintains this itself and this becomes a no-op.
        """
        curvature = self.backbone.manifold_in.c
        for parameter in self.manifold_parameters():
            space = parameter[..., 1:]
            time = (space.square().sum(-1, keepdim=True) + curvature).sqrt()
            parameter.copy_(torch.cat([time, space], dim=-1))

    def manifold_violation(self) -> torch.Tensor:
        """Max ``|<x, x>_L + c|`` over manifold parameters. Should be ~0."""
        worst = torch.zeros(())
        curvature = self.backbone.manifold_in.c
        for parameter in self.manifold_parameters():
            squared = parameter.detach() ** 2
            quad = -squared[..., :1] + squared[..., 1:].sum(-1, keepdim=True)
            worst = torch.maximum(worst, (quad + curvature).abs().max().cpu())
        return worst
