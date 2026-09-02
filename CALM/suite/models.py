"""The four cells of the 2x2, matched so they differ in one thing only.

``README.md`` section 2 sets out why the question is an interaction and needs
all four cells. This module builds them and, more importantly, makes the
matching auditable: :func:`match_euclidean_width` searches for the Euclidean
width whose trainable parameter count lands within tolerance of HELM's, and
:func:`describe` reports total parameters, *active* parameters per token and a
FLOP estimate side by side, because a parameter match and a compute match are
different claims and the reader needs both.

HELM's MoE routes only ``n_activated_experts`` of ``n_routed_experts``, so a
dense Euclidean model matched on total parameters uses **more** compute per
token, while one matched on active parameters is **smaller**. Neither match is
the honest one on its own; reporting both is.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from helm_calm import CalmEnergyHead, HelmCALM, energy_score  # noqa: E402

CELLS = ("helm_discrete", "euclid_discrete", "helm_calm", "euclid_calm")


# --------------------------------------------------------- Euclidean backbone

class EuclideanBackbone(nn.Module):
    """A pre-norm transformer, the control for the hyperbolic backbone.

    Deliberately plain: RMSNorm, multi-head causal attention, SwiGLU. It is not
    trying to be a good model, it is trying to be the *same* model without the
    manifold, so that a difference between the columns is attributable to
    geometry and not to an architectural flourish on one side.

    Accepts pre-embedded input so the CALM cells can hand it patch vectors,
    matching how :class:`~CALM.helm_calm.HelmCALM` feeds its own backbone.
    """

    class Layer(nn.Module):
        def __init__(self, dim: int, heads: int, inter: int):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.RMSNorm(dim, eps=1e-5)
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)
            self.norm2 = nn.RMSNorm(dim, eps=1e-5)
            self.gate = nn.Linear(dim, inter, bias=False)
            self.up = nn.Linear(dim, inter, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, n, d = x.shape
            q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
            shape = (b, n, self.heads, d // self.heads)
            q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.out(attn.transpose(1, 2).reshape(b, n, d))
            h = self.norm2(x)
            return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int,
                 inter: int, max_seq_len: int = 2048):
        super().__init__()
        if dim % heads:
            divisors = [h for h in range(1, dim + 1) if dim % h == 0]
            heads = min(divisors, key=lambda h: (abs(h - heads), h))
        self.dim = dim
        self.heads = heads
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.layers = nn.ModuleList(
            [self.Layer(dim, heads, inter) for _ in range(layers)])
        self.norm = nn.RMSNorm(dim, eps=1e-5)

    def forward(self, x: torch.Tensor, embedded: bool = False) -> torch.Tensor:
        if not embedded:
            x = self.embed(x)
        x = x + self.pos[:, :x.size(1)]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EuclideanDiscrete(nn.Module):
    """Cell ``euclid_discrete``: the baseline HELM's paper is measured against."""

    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int,
                 inter: int, max_seq_len: int = 2048):
        super().__init__()
        self.backbone = EuclideanBackbone(vocab_size, dim, layers, heads, inter,
                                          max_seq_len)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(tokens))

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.logits(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tokens[:, 1:].reshape(-1))


class EuclideanCalm(nn.Module):
    """Cell ``euclid_calm``: CALM as published, with no manifold anywhere.

    Mirrors :class:`~CALM.helm_calm.HelmCALM` stage for stage -- patch
    compression at CALM's own depth, backbone, CALM's generative head, the same
    frozen autoencoder -- so the only difference from ``helm_calm`` is the
    geometry of the backbone and the patch embedding.
    """

    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int,
                 inter: int, patch_size: int, latent_size: int,
                 max_seq_len: int = 2048, num_samples: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_samples = num_samples
        self.embed = nn.Embedding(vocab_size, dim)
        # CALM's embed_proj: Linear -> SiLU -> Linear -> LayerNorm.
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size * dim, 2 * dim), nn.SiLU(),
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim, eps=1e-6))
        self.backbone = EuclideanBackbone(vocab_size, dim, layers, heads, inter,
                                          max_seq_len)
        del self.backbone.embed          # patches are fed in pre-embedded
        self.head = CalmEnergyHead(dim, latent_size)

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(tokens)
        b, s, _ = embedded.shape
        patched = self.patch_embed(embedded.reshape(b, s // self.patch_size, -1))
        return self.backbone(patched, embedded=True)

    def aligned(self, tokens: torch.Tensor):
        n_patches = tokens.size(1) // self.patch_size
        if n_patches < 2:
            raise ValueError("need at least two patches to form a prediction")
        inputs = tokens[:, :(n_patches - 1) * self.patch_size]
        targets = tokens[:, self.patch_size:n_patches * self.patch_size]
        return inputs, targets.reshape(-1, self.patch_size)

    def loss(self, tokens: torch.Tensor, autoencoder) -> torch.Tensor:
        inputs, targets = self.aligned(tokens)
        with torch.no_grad():
            mean, log_std = autoencoder.encode(targets)
        hidden = self.hidden_states(inputs).reshape(-1, self.dim)
        samples = self.head(hidden.unsqueeze(0).expand(self.num_samples, -1, -1))
        return -energy_score(samples, mean, log_std).mean()

    @torch.no_grad()
    def draw(self, tokens: torch.Tensor, autoencoder, n: int):
        """``(n, B, S')`` byte draws plus targets, the shape the HELM cell gives."""
        inputs, targets = self.aligned(tokens)
        hidden = self.hidden_states(inputs).reshape(-1, self.dim)
        latents = self.head(hidden.unsqueeze(0).expand(n, -1, -1))
        decoded = autoencoder.decode(latents).argmax(-1)
        rows = tokens.size(0)
        return decoded.reshape(n, rows, -1), targets.reshape(rows, -1)


# ------------------------------------------------------------------ matching

@dataclass
class Budget:
    """What a cell costs, in the three ways that are not interchangeable."""
    total: int
    active: int
    flops_per_token: float
    detail: Dict[str, int] = field(default_factory=dict)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def helm_active_fraction(args) -> float:
    """Share of MoE expert parameters a single token actually routes to.

    HELM's MiCE activates ``n_activated_experts`` of ``n_routed_experts`` plus
    the shared experts, so total and active parameter counts diverge. Matching
    on one and reporting only that would flatter whichever side the choice
    favours.
    """
    routed = max(getattr(args, "n_routed_experts", 0), 1)
    activated = min(getattr(args, "n_activated_experts", routed), routed)
    shared = getattr(args, "n_shared_experts", 0)
    return (activated + shared) / (routed + shared)


def match_euclidean_width(build_helm: Callable[[], nn.Module],
                          build_euclid: Callable[[int], nn.Module],
                          low: int = 16, high: int = 4096,
                          tolerance: float = 0.02) -> Tuple[int, int, int]:
    """Binary-search the Euclidean width that matches HELM's parameter count.

    Returns ``(width, euclidean_parameters, helm_parameters)``. Raises if no
    width lands within ``tolerance``, rather than silently returning the closest
    -- an unmatched comparison should fail loudly, not quietly become a capacity
    comparison.
    """
    target = count_parameters(build_helm())
    best = None
    while low <= high:
        mid = (low + high) // 2
        count = count_parameters(build_euclid(mid))
        if best is None or abs(count - target) < abs(best[1] - target):
            best = (mid, count)
        if count < target:
            low = mid + 1
        elif count > target:
            high = mid - 1
        else:
            break
    width, count = best
    if abs(count - target) / target > tolerance:
        raise ValueError(
            f"no Euclidean width matches HELM's {target:,} parameters within "
            f"{tolerance:.0%}; closest is width {width} at {count:,} "
            f"({(count - target) / target:+.1%}). Widen the search or relax the "
            f"tolerance deliberately -- do not compare unmatched models.")
    return width, count, target


def describe(model: nn.Module, args=None, seq_len: int = 128,
             is_helm: bool = False) -> Budget:
    """Total parameters, active parameters per token, and a FLOP estimate.

    The FLOP figure is a forward-pass estimate -- ``2 * active_parameters`` per
    token plus attention's ``2 * seq_len * dim`` term -- not an instrumented
    count. It is here to make a compute mismatch visible, not to be quoted.
    """
    total = count_parameters(model)
    active = total
    detail = {}
    if is_helm and args is not None:
        expert = sum(p.numel() for name, p in model.named_parameters()
                     if "experts" in name and p.requires_grad)
        fraction = helm_active_fraction(args)
        active = total - expert + int(expert * fraction)
        detail = {"expert_parameters": expert,
                  "active_expert_fraction": round(fraction, 4)}
    dim = getattr(args, "dim", None) or getattr(model, "dim", 0)
    flops = 2.0 * active + 2.0 * seq_len * float(dim or 0)
    return Budget(total=total, active=active, flops_per_token=flops, detail=detail)


def build_helm_discrete(args, manifolds=None) -> nn.Module:
    manifolds = manifolds or (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    return HelmMiCE(args, *manifolds)


def build_helm_calm(args, autoencoder, manifolds=None, num_samples: int = 8):
    manifolds = manifolds or (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    return HelmCALM(args, autoencoder, manifolds=manifolds,
                    num_samples=num_samples, head_kind="lorentz")
