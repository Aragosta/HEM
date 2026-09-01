"""Hyperbolic Multi-head Latent Attention (HMLA), optimized.

The published HMLA scores two points on the Lorentz manifold by (the negative
of) their squared Lorentzian distance,

    s_ij = 2c + 2 <q_i, k_j>_L ,   <a, b>_L = -a_0 b_0 + a_1..d · b_1..d

then divides by a learned temperature, adds a scalar bias, masks, softmaxes, and
finally aggregates the values with a Lorentzian centroid (a weighted mean
followed by a renormalisation back onto the hyperboloid).

Upstream implements that literally, which forces the full ``(B, H, N, N)`` score
matrix into memory -- twice, since the softmax is taken in fp32 -- and makes the
attention O(N^2) in *memory*, not just in FLOPs. At the released 120M shape
(B=4, H=6, N=2048) that is ~400 MB of activations per layer that also have to be
kept alive for the backward pass.

The key observation is that the whole thing is an ordinary softmax attention in
disguise:

* ``2c`` and ``bias`` are constants along the softmax axis, so they cancel
  exactly. They contribute nothing -- not even a gradient path -- and can be
  dropped.
* Flipping the sign of the time coordinate of the query turns the Minkowski
  inner product into a Euclidean one:
  ``<q, k>_L = (-q_0, q_1..d) · (k_0, k_1..d)``.
* The centroid's numerator ``softmax(s) @ v`` is exactly the attention output
  with the *projected* (time coordinate included) values as ``V``. The
  renormalisation is a cheap elementwise epilogue applied afterwards.

So the whole score-and-aggregate block collapses to a single
``scaled_dot_product_attention`` call and can ride on FlashAttention: no
materialised score matrix, no fp32 copy of it, and free causal masking. The
temperature is folded into the query so that PyTorch's ``scale`` argument stays
a plain float (it cannot take a learnable tensor).

Parameter names and shapes are unchanged, so checkpoints are interchangeable
with the upstream implementation in :mod:`helm.reference.hmla`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from helm.hypercore.nn.conv.conv_util_layers import LorentzRMSNorm
from helm.hypercore.nn.linear.lorentz_linear import LorentzLinear

from .rope import apply_rotary_emb, apply_rotary_emb_real


class LorentzLatentKVCache:
    """MLA cache: the compressed latent plus the decoupled rotary key.

    This is the point of Multi-head *Latent* Attention. The keys and values of
    every head are reconstructed on demand from one shared low-rank latent
    ``c_KV`` (``kv_lora_rank`` wide) and a single rotary key ``k_pe`` shared
    across heads, so what has to be stored per token is those two vectors rather
    than a full per-head K and V.

    Per token, per layer, in elements:

    ===============  ==================================  ==========  =========
    shape            stored                              latent      naive
    ===============  ==================================  ==========  =========
    120M             ``r=65, rope=17, H=6, qk=50, v=33``  **81**      498
    1B               ``r=257, rope=65, H=14, qk=194,      **321**     4522
                     v=129``
    ===============  ==================================  ==========  =========

    -- 6.1x and 14.1x less cache memory respectively. The trade is that
    ``wkv_b`` has to be re-applied to the cached latent each step, which is
    linear in context length; :class:`LorentzKVCache` stores the reconstructed
    keys and values instead if you would rather spend the memory.
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, kv_lora_rank: int,
                 rope_dim: int, dtype: torch.dtype, device: torch.device):
        self.kv = torch.zeros(max_batch_size, max_seq_len, kv_lora_rank,
                              dtype=dtype, device=device)
        self.pe = torch.zeros(max_batch_size, max_seq_len, rope_dim,
                              dtype=dtype, device=device)

    def update(self, kv: torch.Tensor, pe: torch.Tensor, start_pos: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append ``(B, S, r)`` latents and ``(B, S, rope)`` rotary keys; return the prefix."""
        bsz, seqlen, _ = kv.shape
        end = start_pos + seqlen
        self.kv[:bsz, start_pos:end] = kv
        self.pe[:bsz, start_pos:end] = pe
        return self.kv[:bsz, :end], self.pe[:bsz, :end]

    def numel(self) -> int:
        """Elements held, for reporting."""
        return self.kv.numel() + self.pe.numel()


class LorentzKVCache:
    """Naive cache: the reconstructed on-manifold keys and values, per head.

    Larger than :class:`LorentzLatentKVCache` by roughly
    ``H(qk + v) / (r + rope)`` -- 6x at the 120M shape, 14x at 1B -- but it skips
    the ``wkv_b`` reconstruction on every decode step. Worth it only for short
    contexts, where the cache is small anyway.
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int,
                 qk_head_dim: int, v_head_dim: int, dtype: torch.dtype,
                 device: torch.device):
        self.k = torch.zeros(max_batch_size, n_heads, max_seq_len, qk_head_dim,
                             dtype=dtype, device=device)
        self.v = torch.zeros(max_batch_size, n_heads, max_seq_len, v_head_dim,
                             dtype=dtype, device=device)

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write ``k``/``v`` (shape ``(B, H, S, D)``) at ``start_pos``; return the prefix."""
        bsz, _, seqlen, _ = k.shape
        end = start_pos + seqlen
        self.k[:bsz, :, start_pos:end] = k
        self.v[:bsz, :, start_pos:end] = v
        return self.k[:bsz, :, :end], self.v[:bsz, :, :end]

    def numel(self) -> int:
        """Elements held, for reporting."""
        return self.k.numel() + self.v.numel()


class LorentzMLA(nn.Module):
    """Hyperbolic Multi-head Latent Attention.

    Args:
        manifold: the Lorentz manifold the inputs live on.
        args: model config (see ``config/args.py``).
        attn_impl: ``"flash"`` for the fused SDPA path (default), or ``"naive"``
            for the literal upstream formulation. Both produce the same result;
            ``"naive"`` exists to make the equivalence testable.
        rope_impl: ``"complex"`` (default, faster in eager) or ``"real"``
            (fusable under ``torch.compile``). See :mod:`helm.modules.rope`.
    """

    def __init__(self, manifold, args, attn_impl: str = "flash", rope_impl: str = "complex"):
        super().__init__()
        if attn_impl not in ("flash", "naive"):
            raise ValueError(f"attn_impl must be 'flash' or 'naive', got {attn_impl!r}")
        if rope_impl not in ("real", "complex"):
            raise ValueError(f"rope_impl must be 'real' or 'complex', got {rope_impl!r}")
        self.attn_impl = attn_impl
        self.rope_impl = rope_impl

        self.manifold = manifold
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = LorentzLinear(self.manifold, self.dim,
                                    self.n_heads * (self.qk_head_dim - 1))
        else:
            self.wq_a = LorentzLinear(self.manifold, self.dim, self.q_lora_rank - 1)
            self.q_norm = LorentzRMSNorm(self.manifold, self.q_lora_rank)
            self.wq_b = LorentzLinear(self.manifold, self.q_lora_rank + 1,
                                      self.n_heads * (self.qk_head_dim - 1))
        self.wkv_a = LorentzLinear(self.manifold, self.dim,
                                   self.kv_lora_rank + self.qk_rope_head_dim - 1)
        self.kv_norm = LorentzRMSNorm(self.manifold, self.kv_lora_rank)
        self.wkv_b = LorentzLinear(self.manifold, self.kv_lora_rank + 1,
                                   self.n_heads * (self.qk_nope_head_dim + self.v_head_dim - 1))
        self.wo = LorentzLinear(manifold, self.n_heads * self.v_head_dim, self.dim - 1)

        # NOTE(upstream bug): the released code does
        #     self.softmax_scale = self.softmax_scale * mscale * mscale
        # which *rebinds* the attribute from an nn.Parameter to a plain
        # non-leaf tensor. The temperature then silently stops being trained and
        # disappears from the state dict. It also reads ``args.mscale``, which
        # ``config/args.py`` never defines. We fold the YaRN correction into the
        # initial value instead, so the parameter stays a parameter.
        scale_init = math.sqrt(self.n_local_heads * self.qk_head_dim)
        if args.max_seq_len > args.original_seq_len:
            mscale_cfg = getattr(args, "mscale", 1.0)
            mscale = 0.1 * mscale_cfg * math.log(args.rope_factor) + 1.0
            scale_init = scale_init * mscale * mscale
        self.softmax_scale = nn.Parameter(torch.tensor([scale_init]))
        # Kept for checkpoint compatibility. It is a scalar added to every score
        # *before* the softmax, so it cancels exactly: it has no effect on the
        # output and its gradient is identically zero. Upstream trains it anyway
        # (and excludes it from weight decay), so it stays pinned at 0 forever.
        # Freezing it is therefore behaviour-preserving, and it keeps DDP from
        # tripping over a parameter that never receives a gradient -- one of the
        # reasons the reference training script has to pay for
        # ``find_unused_parameters=True``.
        self.bias = nn.Parameter(torch.zeros(()), requires_grad=False)

    # ------------------------------------------------------------------ utils

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Lift a space-like vector onto the hyperboloid by solving for x_0 > 0."""
        x_time = (x.square().sum(dim=-1, keepdim=True) + self.manifold.c).clamp_min(1e-12).sqrt()
        return torch.cat([x_time, x], dim=-1)

    def project_neg_time(self, x: torch.Tensor) -> torch.Tensor:
        """Same lift, but with the time coordinate negated.

        ``project_neg_time(q) · project(k)`` (a plain Euclidean dot product) is
        the Minkowski inner product ``<q, k>_L``, which is what lets the scores
        go through ``scaled_dot_product_attention``.
        """
        x_time = (x.square().sum(dim=-1, keepdim=True) + self.manifold.c).clamp_min(1e-12).sqrt()
        return torch.cat([-x_time, x], dim=-1)

    @staticmethod
    def shape_mask(mask: Optional[torch.Tensor], batch_size: int, num_heads: int,
                   seq_len: int) -> Optional[torch.Tensor]:
        """Broadcast an attention mask to ``(B|1, H|1, N, N)``. ``True`` == masked."""
        if mask is None:
            return None
        if mask.dim() == 4:
            return mask
        if mask.dim() == 2:
            m, n = mask.shape
            if m == seq_len and n == seq_len:          # [N, N]
                return mask.unsqueeze(0).unsqueeze(0)
            return mask.unsqueeze(1).unsqueeze(2)      # [B, N] -> [B, 1, 1, N]
        if mask.dim() == 3:
            b, m, n = mask.shape
            if m == seq_len and n == seq_len:          # [B, N, N]
                return mask.unsqueeze(1)
            if m == 1 and n == seq_len:                # [B, 1, N]
                return mask.squeeze(1).unsqueeze(1).unsqueeze(2)
            if m == num_heads and n == seq_len:        # [B, H, N]
                return mask.unsqueeze(2)
        raise ValueError(f"unsupported attention-mask shape {tuple(mask.shape)}")

    def _centroid(self, ctx: torch.Tensor) -> torch.Tensor:
        """Renormalise a convex combination of hyperboloid points back onto it."""
        d = ctx.size(-1) - 1
        sq = ctx.square()
        # <ctx, ctx>_L, computed inline to avoid a second pass over the tensor.
        inner = -sq.narrow(-1, 0, 1) + sq.narrow(-1, 1, d).sum(dim=-1, keepdim=True)
        denom = inner.neg().abs().clamp_min(self.manifold.eps[ctx.dtype]).sqrt()
        return self.manifold.c.sqrt() * ctx / denom

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor, start_pos: int = 0,
                freqs_cis: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                cache: Optional[LorentzKVCache] = None) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, dim)`` on the manifold.
            start_pos: offset of this chunk into the sequence (for the cache).
            freqs_cis: rotary table slice; complex if ``rope_impl == "complex"``,
                otherwise the real ``(cos, sin)`` table.
            mask: ``True`` == masked. Any shape :meth:`shape_mask` accepts.
            is_causal: apply a causal mask without materialising one. Mutually
                exclusive with ``mask``.
            cache: :class:`LorentzLatentKVCache` (compressed, the MLA way) or
                :class:`LorentzKVCache` (reconstructed keys/values). Either way
                attention runs over the whole cached prefix.

        Returns:
            ``(B, N, dim)`` on the manifold.
        """
        if is_causal and mask is not None:
            raise ValueError("pass either `mask` or `is_causal=True`, not both")
        bsz, seqlen, _ = x.size()

        if self.q_lora_rank == 0:
            q = self.wq(x, return_space=True)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x, return_space=True), space_only=True),
                          return_space=True)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim - 1)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim - 1], dim=-1)

        rope = apply_rotary_emb_real if self.rope_impl == "real" else apply_rotary_emb
        q_pe = rope(q_pe, freqs_cis)

        kv = self.wkv_a(x, return_space=True)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim - 1], dim=-1)
        k_pe = rope(k_pe.unsqueeze(2), freqs_cis)

        q = torch.cat([q_nope, q_pe], dim=-1)

        latent_cache = isinstance(cache, LorentzLatentKVCache)
        if latent_cache:
            # Cache before reconstruction: this is what makes the cache latent.
            kv, pe_all = cache.update(kv, k_pe.squeeze(2), start_pos)
            k_pe = pe_all.unsqueeze(2)

        kv = self.wkv_b(self.kv_norm(kv, space_only=True), return_space=True)
        kv = kv.view(bsz, kv.size(1), self.n_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim - 1)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim - 1], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

        naive_cache = cache if isinstance(cache, LorentzKVCache) else None
        if self.attn_impl == "flash":
            out = self._attend_flash(q, k, v, bsz, seqlen, mask, is_causal, start_pos,
                                     naive_cache)
        else:
            out = self._attend_naive(q, k, v, bsz, seqlen, mask, is_causal, start_pos,
                                     naive_cache)
        return self.wo(out.flatten(2))

    def _attend_flash(self, q, k, v, bsz, seqlen, mask, is_causal, start_pos, cache):
        # (B, N, H, D) -> (B, H, N, D) for SDPA.
        # The 2/temperature factor is folded into the query so `scale` can stay a
        # plain float; `softmax_scale` is a learnable tensor and SDPA's `scale`
        # argument only accepts a Python float.
        qs = self.project_neg_time(q).transpose(1, 2) * (2.0 / self.softmax_scale)
        ks = self.project(k).transpose(1, 2)
        vs = self.project(v).transpose(1, 2)

        if cache is not None:
            ks, vs = cache.update(ks, vs, start_pos)

        attn_mask = None
        if mask is not None:
            # SDPA's boolean convention is the opposite of ours: True == attend.
            # Every row is guaranteed at least one unmasked entry (the diagonal
            # survives both the causal mask and the same-document mask), so this
            # cannot produce an all-masked row and the NaN that would follow.
            attn_mask = ~self.shape_mask(mask, bsz, self.n_local_heads, seqlen)

        ctx = F.scaled_dot_product_attention(
            qs, ks, vs, attn_mask=attn_mask, is_causal=is_causal, scale=1.0,
        )
        return self._centroid(ctx).transpose(1, 2)

    def _attend_naive(self, q, k, v, bsz, seqlen, mask, is_causal, start_pos, cache):
        """Literal upstream formulation. Kept as the reference for the fast path."""
        qs = self.project(q).transpose(1, 2)
        ks = self.project(k).transpose(1, 2)
        vs = self.project(v).transpose(1, 2)
        if cache is not None:
            ks, vs = cache.update(ks, vs, start_pos)

        d = qs.size(-1) - 1
        qs_flip = torch.cat([-qs.narrow(-1, 0, 1), qs.narrow(-1, 1, d)], dim=-1)
        scores = 2 * self.manifold.c + 2 * (qs_flip @ ks.transpose(-1, -2))
        scores = scores / self.softmax_scale + self.bias

        if is_causal:
            kv_len = ks.size(-2)
            causal = torch.ones(seqlen, kv_len, dtype=torch.bool, device=scores.device)
            causal = causal.triu(kv_len - seqlen + 1)
            scores = scores.masked_fill(causal, -1e18)
        elif mask is not None:
            scores = scores.masked_fill(
                self.shape_mask(mask, bsz, self.n_local_heads, seqlen), -1e18)

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(qs)
        return self._centroid(scores @ vs).transpose(1, 2)
