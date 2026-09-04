"""A Euclidean MoE+MHA decoder with hyperbolic geometry only where it can pay.

`WHY_HYPERBOLIC.md` sets out the argument. In one line: on the FlyWire
connectome a 2D hyperbolic embedding beats the fly's own 3D anatomy, but
Euclidean overtakes it between d=8 and d=16 and wins outright by d=32-128. So
hyperbolic geometry buys **dimension efficiency**, and a 2048-8192 wide residual
stream is 100-1000x past the point where that matters. HELM applies it there
anyway, and in our controlled T0 lost 3.1x on perplexity.

A decoder's *low-dimensional* parts are a different regime:

* **per-head attention space**, d_head = 16-128 -- squarely in the window;
* **the MLA/KV latent**, which exists precisely to be small.

And for the first there is a theorem rather than an analogy: the Modern Hopfield
update rule *is* attention (Ramsauer et al. 2020), and in hyperbolic space it
gains a double-exponential capacity term absent from Euclidean associative
memory, with gains "most pronounced when the hidden dimensionality is
constrained" (arXiv:2606.10238).

**Why this implementation cannot suffer the usual cancellation.** The standard
critique of hyperbolic networks is that chained exp/log maps are mutually
inverse and collapse to the Euclidean transform. Here there are no exp/log maps
at all. A head vector ``q`` is lifted to the hyperboloid by appending the time
coordinate ``sqrt(|q|^2 + c)``, and the Lorentz inner product

    <q, k>_L = -q_t k_t + q . k

is then **a plain dot product of (d_head + 1)-vectors with the query's time
coordinate negated**. So the whole geometry is: one elementwise sqrt per vector,
and one extra dimension in the existing matmul. Same kernel, same asymptotics --
which is the "same computational complexity" the theorem relies on.

Everything else -- residual stream, FFN, MoE routing, embeddings, norms -- is
ordinary Euclidean, unchanged between arms, so a difference is attributable to
the head geometry alone.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

GEOMETRIES = ("euclidean", "lorentz")


def lift_to_hyperboloid(x: torch.Tensor, c: float = 1.0,
                        negate_time: bool = False) -> torch.Tensor:
    """``(..., d)`` -> ``(..., d + 1)`` on the hyperboloid of curvature ``-1/c``.

    With ``negate_time=True`` the time coordinate is negated, which turns the
    Minkowski inner product into an ordinary dot product:

        <q, k>_L = -q_t k_t + q.k = lift(q, negate)  .  lift(k)

    That identity is what lets the whole thing run through
    ``scaled_dot_product_attention`` unchanged -- the geometry costs one sqrt
    and one extra dimension, not a new kernel.
    """
    time = (x.square().sum(-1, keepdim=True) + c).clamp_min(1e-8).sqrt()
    lifted = torch.cat([-time if negate_time else time, x], dim=-1)
    if lifted.shape[-1] % 2:
        # Pad to an even width with a zero. It contributes 0 to every inner
        # product, so the geometry is unchanged -- but an odd head dimension
        # knocks scaled_dot_product_attention off its fast path, which measured
        # as a 37% slowdown at head_dim 32 against 1.8% at 16. That is an
        # artefact of kernel dispatch, not a cost of the geometry, and letting
        # it stand would have made the hyperbolic arm look expensive for the
        # wrong reason.
        lifted = F.pad(lifted, (0, 1))
    return lifted


class Attention(nn.Module):
    """Multi-head attention with an optional MLA-style latent and a head geometry.

    Args:
        dim: residual width (always Euclidean).
        heads: number of heads.
        head_dim: per-head width -- **the variable under test**.
        kv_latent: when set, keys and values are routed through a latent of this
            width, as in DeepSeek's MLA. ``None`` gives ordinary MHA.
        head_geometry: ``"euclidean"`` for the dot product, ``"lorentz"`` for the
            Minkowski inner product on the lifted head vectors.
        latent_geometry: same, applied to the KV latent.
    """

    def __init__(self, dim: int, heads: int, head_dim: int,
                 kv_latent: Optional[int] = None,
                 head_geometry: str = "euclidean",
                 latent_geometry: str = "euclidean",
                 curvature: float = 1.0, max_seq_len: int = 2048):
        super().__init__()
        for name, value in (("head_geometry", head_geometry),
                            ("latent_geometry", latent_geometry)):
            if value not in GEOMETRIES:
                raise ValueError(f"{name} must be one of {GEOMETRIES}, got {value!r}")
        self.heads, self.head_dim = heads, head_dim
        self.head_geometry, self.latent_geometry = head_geometry, latent_geometry
        self.c = curvature
        self.kv_latent = kv_latent
        inner = heads * head_dim

        self.wq = nn.Linear(dim, inner, bias=False)
        if kv_latent is None:
            self.wk = nn.Linear(dim, inner, bias=False)
            self.wv = nn.Linear(dim, inner, bias=False)
        else:
            self.w_down = nn.Linear(dim, kv_latent, bias=False)
            self.latent_norm = nn.RMSNorm(kv_latent, eps=1e-5)
            # The latent carries its time coordinate into the up-projection when
            # it is hyperbolic, so the up-projection is one wider. That is the
            # only parameter-count difference the geometry introduces anywhere.
            up_in = kv_latent + (1 if latent_geometry == "lorentz" else 0)
            self.w_up_k = nn.Linear(up_in, inner, bias=False)
            self.w_up_v = nn.Linear(up_in, inner, bias=False)
        self.wo = nn.Linear(inner, dim, bias=False)

        # Rotary, applied in the *Euclidean* head coordinates before the lift.
        # Rotation is an isometry of the space part and leaves |q| unchanged, so
        # the time coordinate -- and hence the geometry -- is unaffected by it.
        half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half).float() / max(half, 1)))
        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
        self.register_buffer("rotary", torch.polar(torch.ones_like(angles), angles),
                             persistent=False)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_dim % 2:
            return x
        shape = x.shape
        paired = torch.view_as_complex(x.float().reshape(*shape[:-1], -1, 2))
        out = paired * self.rotary[: shape[-2]].view(1, 1, shape[-2], -1)
        return torch.view_as_real(out).reshape(*shape).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        shape = (b, n, self.heads, self.head_dim)
        q = self.wq(x).view(shape).transpose(1, 2)

        if self.kv_latent is None:
            k = self.wk(x).view(shape).transpose(1, 2)
            v = self.wv(x).view(shape).transpose(1, 2)
        else:
            latent = self.latent_norm(self.w_down(x))
            if self.latent_geometry == "lorentz":
                latent = lift_to_hyperboloid(latent, self.c)
            k = self.w_up_k(latent).view(shape).transpose(1, 2)
            v = self.w_up_v(latent).view(shape).transpose(1, 2)

        q, k = self._rotate(q), self._rotate(k)

        if self.head_geometry == "lorentz":
            # <q,k>_L as a plain dot product on (head_dim + 1)-vectors. The
            # score is the Minkowski inner product, which is a monotone function
            # of the negative squared Lorentz distance -- the constant offset
            # 2c that separates them is invariant under softmax.
            q = lift_to_hyperboloid(q, self.c, negate_time=True)
            k = lift_to_hyperboloid(k, self.c)
            scale = 1.0 / math.sqrt(self.head_dim + 1)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        return self.wo(out.transpose(1, 2).reshape(b, n, -1))


class MoE(nn.Module):
    """Top-k routed SwiGLU experts plus a shared expert. Euclidean throughout."""

    class Expert(nn.Module):
        def __init__(self, dim: int, inter: int):
            super().__init__()
            self.gate = nn.Linear(dim, inter, bias=False)
            self.up = nn.Linear(dim, inter, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)

        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    def __init__(self, dim: int, inter: int, n_experts: int = 4, top_k: int = 2,
                 n_shared: int = 1):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [self.Expert(dim, inter) for _ in range(n_experts)])
        self.shared = nn.ModuleList(
            [self.Expert(dim, inter) for _ in range(n_shared)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        flat = x.reshape(-1, d)
        scores = self.router(flat).softmax(-1)
        weight, index = scores.topk(self.top_k, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            for expert_id, expert in enumerate(self.experts):
                mask = index[:, slot] == expert_id
                if mask.any():
                    out[mask] += weight[mask, slot, None] * expert(flat[mask])
        for expert in self.shared:
            out = out + expert(flat)
        return out.reshape(b, n, d)


class Block(nn.Module):
    def __init__(self, dim, heads, head_dim, inter, kv_latent, head_geometry,
                 latent_geometry, max_seq_len, dense: bool):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-5)
        self.attn = Attention(dim, heads, head_dim, kv_latent, head_geometry,
                              latent_geometry, max_seq_len=max_seq_len)
        self.norm2 = nn.RMSNorm(dim, eps=1e-5)
        self.ffn = (MoE.Expert(dim, inter) if dense
                    else MoE(dim, inter // 2))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class HybridDecoder(nn.Module):
    """The model under test. Euclidean everywhere except the two named places."""

    def __init__(self, vocab: int, dim: int = 256, layers: int = 6,
                 heads: int = 8, head_dim: int = 32, inter: Optional[int] = None,
                 kv_latent: Optional[int] = 64,
                 head_geometry: str = "euclidean",
                 latent_geometry: str = "euclidean",
                 max_seq_len: int = 512, n_dense: int = 1):
        super().__init__()
        inter = inter or 4 * dim
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            Block(dim, heads, head_dim, inter, kv_latent, head_geometry,
                  latent_geometry, max_seq_len, dense=(i < n_dense))
            for i in range(layers)])
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight          # tied
        self.apply(self._init)
        # Scale the residual-path output projections by 1/sqrt(2 * layers), the
        # usual GPT-2 rule, so the residual stream does not grow with depth.
        scale = (2 * layers) ** -0.5
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.wo.weight, std=0.02 * scale)
            for module in block.ffn.modules():
                if isinstance(module, nn.Linear) and module.out_features == dim:
                    torch.nn.init.normal_(module.weight, std=0.02 * scale)

    @staticmethod
    def _init(module):
        """std=0.02 everywhere.

        PyTorch initialises nn.Embedding from N(0, 1). With tied weights that
        makes the logits' standard deviation scale with sqrt(dim), and the
        initial loss came out at 237 against ln(16000) = 9.68 -- a model so
        badly conditioned that any architecture comparison built on it would be
        measuring the initialisation.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.logits(tokens[:, :-1])
        return F.cross_entropy(out.reshape(-1, out.size(-1)),
                               tokens[:, 1:].reshape(-1))
