"""Rotary positional embeddings for HELM-MiCE.

Two interchangeable implementations of the *same* rotation:

``apply_rotary_emb``
    The upstream ``torch.view_as_complex`` formulation, kept verbatim as the
    reference.

``apply_rotary_emb_real``
    A real-arithmetic equivalent driven by a precomputed ``(cos, sin)`` table.

The real path is preferred because the complex one is expensive and opaque:
``x.float().contiguous()`` materialises a full fp32 copy of the tensor, the
complex multiply is a separate kernel, and ``view_as_real(...).flatten(3)``
forces yet another copy before the cast back to bf16 -- three full round trips
through memory for what is six elementwise ops. More importantly, TorchInductor
has no lowering for complex tensors, so a single ``view_as_complex`` in the
attention block silently forces a graph break and de-optimises the whole layer
under ``torch.compile``. The real formulation fuses into one kernel.

Both use the *interleaved* pair convention -- ``(x[..., 0], x[..., 1])`` form the
first complex number, ``(x[..., 2], x[..., 3])`` the second -- because that is
what ``view_as_complex`` does, and the two must stay bit-compatible so that a
checkpoint trained under one runs under the other.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def precompute_freqs_cis(args) -> torch.Tensor:
    """Precompute the YaRN-scaled complex rotation table. (Upstream, verbatim.)

    Returns:
        Complex tensor of shape ``(max_seq_len, (qk_rope_head_dim - 1) // 2)``.
    """
    dim = args.qk_rope_head_dim - 1
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_rope_cache(args) -> torch.Tensor:
    """Real-valued twin of :func:`precompute_freqs_cis`.

    Returns:
        Float tensor of shape ``(max_seq_len, (qk_rope_head_dim - 1) // 2, 2)``
        holding ``(cos, sin)`` in the last dimension. Stored as one tensor so it
        registers as a single buffer and slices with one indexing op.
    """
    freqs_cis = precompute_freqs_cis(args)
    return torch.stack((freqs_cis.real, freqs_cis.imag), dim=-1).contiguous()


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Reference (complex) rotary embedding, as published.

    Args:
        x: ``(batch, seqlen, heads, head_dim)``, ``head_dim`` even.
        freqs_cis: complex ``(seqlen, head_dim // 2)``.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().contiguous().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


def apply_rotary_emb_real(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """Real-arithmetic rotary embedding. Numerically equal to the complex path.

    Args:
        x: ``(batch, seqlen, heads, head_dim)``, ``head_dim`` even.
        rope_cache: ``(seqlen, head_dim // 2, 2)`` of ``(cos, sin)``.
    """
    dtype = x.dtype
    seqlen = x.size(1)
    # (B, S, H, D/2, 2) -- a view, no copy.
    xf = x.float().unflatten(-1, (-1, 2))
    x0 = xf[..., 0]
    x1 = xf[..., 1]
    # (1, S, 1, D/2) so it broadcasts over batch and heads.
    cos = rope_cache[..., 0].view(1, seqlen, 1, -1)
    sin = rope_cache[..., 1].view(1, seqlen, 1, -1)
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.flatten(-2).to(dtype)
