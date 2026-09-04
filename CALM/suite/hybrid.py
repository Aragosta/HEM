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
BETA_MODES = ("fixed", "learned", "logn")
NORMALIZERS = ("softmax", "entmax")
FFN_KINDS = ("dense", "moe")


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
                 curvature: float = 1.0, max_seq_len: int = 2048,
                 beta_mode: str = "fixed", ref_len: int = 128,
                 normalizer: str = "softmax", alpha_init: float = 0.0):
        super().__init__()
        if beta_mode not in BETA_MODES:
            raise ValueError(f"beta_mode must be one of {BETA_MODES}, got {beta_mode!r}")
        if normalizer not in NORMALIZERS:
            raise ValueError(f"normalizer must be one of {NORMALIZERS}, got {normalizer!r}")
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

        # --- inverse temperature (CRITICALITY.md sec.1) ---------------------
        # A score matrix passed through softmax is a dense graph annealed at
        # inverse temperature beta. The default beta = 1/sqrt(d) is a variance
        # argument, not an optimality argument, and the phase-transition result
        # says the useful regime is a narrow band. So make beta a first-class
        # knob instead of a constant folded into the kernel call.
        #
        # For the Lorentz arm the effective width is head_dim + 1, so the
        # *default* beta already differs between geometries -- which is exactly
        # the confound C1 suspects in T0. Making it learnable lets each arm find
        # its own temperature rather than inheriting one from the other's algebra.
        eff = head_dim + 1 if head_geometry == "lorentz" else head_dim
        self.base_scale = 1.0 / math.sqrt(eff)
        self.beta_mode, self.ref_len = beta_mode, ref_len
        if beta_mode == "learned":
            # One scalar per head, parameterised in log space so it stays
            # positive. Initialised at the standard value, so at step 0 the
            # learned arm is bit-identical to the fixed arm.
            self.log_beta = nn.Parameter(
                torch.full((heads, 1, 1), math.log(self.base_scale)))
        # --- learned sparsity (alpha-entmax) --------------------------------
        # alpha-entmax(z) = argmax_p <p,z> + H^T_alpha(p) over the simplex,
        # with H^T the Tsallis entropy family. alpha = 1 recovers softmax
        # exactly; alpha = 2 is sparsemax; alpha in (1,2) interpolates, and for
        # every alpha > 1 the solution has EXACT ZEROS.
        #
        # This is the part `beta` could not reach. A temperature rescales the
        # logits but softmax always has full support -- every token keeps a
        # nonzero weight. alpha changes the *support*: it learns which tokens
        # to drop. So T2's null result on beta says nothing about this; they
        # are different mechanisms (temperature vs topology).
        #
        # Parameterised as alpha = 1 + sigmoid(a), one scalar per head, per
        # Correia, Niculae & Martins (arXiv:1909.00015), who derive the
        # Jacobian w.r.t. alpha that makes the sparsity level itself learnable.
        self.normalizer = normalizer
        if normalizer == "entmax":
            self.alpha_logit = nn.Parameter(torch.full((heads, 1, 1), alpha_init))
        self.collect_stats = False
        self.stats: dict = {}

        # Rotary, applied in the *Euclidean* head coordinates before the lift.
        # Rotation is an isometry of the space part and leaves |q| unchanged, so
        # the time coordinate -- and hence the geometry -- is unaffected by it.
        half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half).float() / max(half, 1)))
        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
        self.register_buffer("rotary", torch.polar(torch.ones_like(angles), angles),
                             persistent=False)

    def _scale(self, n: int):
        """Return the attention scale: a float, or a per-head tensor."""
        if self.beta_mode == "learned":
            return self.log_beta.exp()
        if self.beta_mode == "logn":
            # beta_n ~ log n (arXiv:2510.05554). Anchored so that at the
            # training length this is exactly the standard scale -- it is
            # therefore a no-op at fixed length and only bites on extrapolation.
            return self.base_scale * math.log(max(n, 2)) / math.log(max(self.ref_len, 2))
        return self.base_scale

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

        # Fold beta into q rather than passing it to SDPA, because a learned
        # beta is one scalar *per head* and SDPA's `scale` takes a float only.
        # q is (b, heads, n, d) and log_beta is (heads, 1, 1), so this
        # broadcasts to a per-head temperature at no extra cost.
        q = q * self._scale(n)

        if self.normalizer == "entmax":
            out = self._attend_entmax(q, k, v, n)
        elif self.collect_stats:
            out = self._attend_with_stats(q, k, v, n)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        return self.wo(out.transpose(1, 2).reshape(b, n, -1))

    def alpha(self):
        return 1.0 + torch.sigmoid(self.alpha_logit)

    def _attend_entmax(self, q, k, v, n):
        """alpha-entmax attention. No fused kernel exists, so this is explicit.

        Masked positions are set to -inf before the transform, as with softmax;
        entmax_bisect maps -inf to exactly zero weight, so causality holds.
        """
        from entmax import entmax_bisect
        scores = q @ k.transpose(-2, -1)
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        alpha = self.alpha()
        p = entmax_bisect(scores, alpha=alpha, dim=-1)

        if self.collect_stats:
            with torch.no_grad():
                allowed = torch.arange(1, n + 1, device=q.device).float()
                part = 1.0 / p.square().sum(-1).clamp_min(1e-12)
                nonzero = (p > 0).float().sum(-1)
                self.stats = {
                    "participation": part[..., 1:].mean().item(),
                    "participation_frac": (part[..., 1:] / allowed[1:]).mean().item(),
                    # The quantity beta could not move: the fraction of
                    # allowed keys that receive EXACTLY zero weight.
                    "zero_frac": (1 - nonzero[..., 1:] / allowed[1:]).mean().item(),
                    "alpha": alpha.mean().item(),
                    "alpha_min": alpha.min().item(),
                    "alpha_max": alpha.max().item(),
                }
        return p @ v

    def _attend_with_stats(self, q, k, v, n):
        """Explicit attention that also records the order parameters.

        Only ever called under `collect_stats`, i.e. at eval on a handful of
        batches, so it does not need to be fast -- but it does need to compute
        *exactly* what the fused path computes, or the diagnostics describe a
        different model than the one being scored. Hence the same causal mask
        and the same already-scaled q.
        """
        scores = q @ k.transpose(-2, -1)
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        p = scores.softmax(-1)

        with torch.no_grad():
            # Participation ratio 1 / sum(p^2): the *effective number of tokens
            # attended*. This is the sharpest read on which phase a head is in.
            # Near n -> disordered (uniform average, rank collapse); near 1 ->
            # frozen (the identity/copy regime); in between is the useful phase.
            allowed = torch.arange(1, n + 1, device=q.device).float()
            part = 1.0 / p.square().sum(-1).clamp_min(1e-12)      # (b, h, n)
            entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
            # Normalise each position by its own maximum: position j may attend
            # to j+1 keys, so max entropy is log(j+1). Position 0 has exactly
            # one choice and is uninformative, so it is dropped rather than
            # divided by log(1) = 0.
            norm_ent = entropy[..., 1:] / allowed[1:].log()
            self.stats = {
                "participation": part[..., 1:].mean().item(),
                "participation_frac": (part[..., 1:] / allowed[1:]).mean().item(),
                "entropy_norm": norm_ent.mean().item(),
                "beta": float(self._scale(n).mean()) if self.beta_mode == "learned"
                        else float(self._scale(n)),
                "beta_ratio": (float(self._scale(n).mean()) / self.base_scale
                               if self.beta_mode == "learned" else 1.0),
            }
        return p @ v


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
        """`inter` is the width of ONE expert, not the layer's total.

        The caller is responsible for matching FLOPs: with `top_k + n_shared`
        experts active per token, a dense FFN of width `W` is matched by experts
        of width `W / (top_k + n_shared)`. `expert_width_for` does that
        arithmetic so a comparison cannot silently drift into being a
        capacity comparison instead of a routing comparison.
        """
        super().__init__()
        self.top_k, self.n_experts = top_k, n_experts
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [self.Expert(dim, inter) for _ in range(n_experts)])
        self.shared = nn.ModuleList(
            [self.Expert(dim, inter) for _ in range(n_shared)])
        self.collect_stats = False
        self.stats: dict = {}

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

        if self.collect_stats:
            with torch.no_grad():
                # Top-k routing is a *discrete* graph reduction over the expert
                # graph, so it has its own order parameters. Router entropy says
                # how close routing is to uniform (no reduction happening) or to
                # a hard one-hot commitment; load imbalance is the collapse mode
                # where reduction has gone too far and experts are dead.
                ent = -(scores * scores.clamp_min(1e-12).log()).sum(-1).mean()
                load = torch.zeros(self.n_experts, device=flat.device)
                load.scatter_add_(0, index.reshape(-1),
                                  torch.ones_like(index.reshape(-1),
                                                  dtype=load.dtype))
                load = load / load.sum().clamp_min(1)
                self.stats = {
                    "router_entropy_norm": (ent / math.log(self.n_experts)).item(),
                    # 1 = perfectly balanced, 1/n_experts = fully collapsed.
                    "load_balance": (1.0 / (self.n_experts
                                            * load.square().sum())).item(),
                    "dead_experts": int((load == 0).sum().item()),
                }
        return out.reshape(b, n, d)


def expert_width_for(dense_inter: int, top_k: int = 2, n_shared: int = 1) -> int:
    """Expert width that makes an MoE layer FLOP-matched to a dense FFN.

    Without this the usual MoE/dense comparison is confounded: the MoE arm gets
    both a routing mechanism *and* more compute per token, and a win cannot be
    attributed to either.
    """
    return max(1, round(dense_inter / (top_k + n_shared)))


class Block(nn.Module):
    def __init__(self, dim, heads, head_dim, inter, kv_latent, head_geometry,
                 latent_geometry, max_seq_len, dense: bool,
                 beta_mode: str = "fixed", top_k: int = 2, n_experts: int = 4,
                 n_shared: int = 1, normalizer: str = "softmax",
                 alpha_init: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-5)
        self.attn = Attention(dim, heads, head_dim, kv_latent, head_geometry,
                              latent_geometry, max_seq_len=max_seq_len,
                              beta_mode=beta_mode, ref_len=max_seq_len,
                              normalizer=normalizer, alpha_init=alpha_init)
        self.norm2 = nn.RMSNorm(dim, eps=1e-5)
        # FLOP-matched: the MoE arm activates top_k + n_shared experts of width
        # inter/(top_k + n_shared), so both arms do the same work per token and
        # differ only in whether that work is routed.
        self.ffn = (MoE.Expert(dim, inter) if dense else
                    MoE(dim, expert_width_for(inter, top_k, n_shared),
                        n_experts=n_experts, top_k=top_k, n_shared=n_shared))

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
                 max_seq_len: int = 512, n_dense: int = 1,
                 ffn: Optional[str] = None, beta_mode: str = "fixed",
                 top_k: int = 2, n_experts: int = 4, n_shared: int = 1,
                 normalizer: str = "softmax", alpha_init: float = 0.0):
        super().__init__()
        if ffn is not None:
            if ffn not in FFN_KINDS:
                raise ValueError(f"ffn must be one of {FFN_KINDS}, got {ffn!r}")
            # `ffn` is the blunt switch used by the dense-vs-MoE comparison:
            # every block is one kind or the other. `n_dense` (leading dense
            # blocks, as in DeepSeek) stays available for anything else.
            n_dense = layers if ffn == "dense" else 0
        inter = inter or 4 * dim
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            Block(dim, heads, head_dim, inter, kv_latent, head_geometry,
                  latent_geometry, max_seq_len, dense=(i < n_dense),
                  beta_mode=beta_mode, top_k=top_k, n_experts=n_experts,
                  n_shared=n_shared, normalizer=normalizer,
                  alpha_init=alpha_init)
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

    def set_stats(self, on: bool) -> None:
        """Turn the order-parameter instrumentation on or off everywhere."""
        for module in self.modules():
            if isinstance(module, (Attention, MoE)):
                module.collect_stats = on
                if not on:
                    module.stats = {}

    def order_parameters(self) -> dict:
        """Mean of each recorded statistic across layers, plus per-layer lists.

        Call after a forward pass taken under `set_stats(True)`.
        """
        collected: dict = {}
        for block in self.blocks:
            for module in (block.attn, block.ffn):
                for key, value in getattr(module, "stats", {}).items():
                    collected.setdefault(key, []).append(value)
        summary = {k: sum(v) / len(v) for k, v in collected.items()}
        summary["per_layer"] = collected
        return summary

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.logits(tokens[:, :-1])
        return F.cross_entropy(out.reshape(-1, out.size(-1)),
                               tokens[:, 1:].reshape(-1))
