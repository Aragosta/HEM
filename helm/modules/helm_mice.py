"""HELM-MiCE: a hyperbolic decoder with latent attention and curvature experts.

This is the optimized assembly of :mod:`helm.modules.hmla` and
:mod:`helm.modules.mice`. Beyond what those two modules do, the changes here are:

* **The causal mask is not materialised** when there is no document mask.
  Upstream always builds an ``(N, N)`` boolean mask and hands it to attention,
  which forces the masked (non-flash) SDPA path; passing ``is_causal=True``
  instead lets FlashAttention skip the entire upper triangle -- roughly half the
  attention FLOPs -- and allocates nothing.
* **The mask is broadcast once**, in the model, to ``(B, 1, N, N)``, instead of
  being re-derived inside every layer.
* **Eval mode works.** Upstream's ``Block.forward`` and ``LorentzMoE.forward``
  disagree about their arity outside training, so the released model raises as
  soon as ``.eval()`` is called (see :class:`helm.modules.mice.Gate`).
* **A fused Lorentz residual** (:class:`helm.modules.lorentz_ops.LorentzResidual`),
  which is where a surprising amount of the non-GEMM time goes.
* **A fused language-model head** (``labels=...``). ``dim=390`` against a
  128256-entry vocabulary makes ``head`` half the parameters and most of the
  forward pass, and ``head(h).float()`` materialises 3.9 GiB of logits at the
  released training shape. See :mod:`helm.modules.fused_ce`.
* **Optional activation checkpointing** (``--grad_checkpoint``), which trades
  ~30% recompute for a large activation saving and usually buys back more than
  it costs by allowing a bigger micro-batch.
* **Incremental decoding** via :class:`helm.modules.hmla.LorentzKVCache`.
  Upstream ships its cache commented out, so generation re-runs the whole prefix
  for every token.
* The ``attn_mask`` buffer is non-persistent, keeping a dense ``max_seq_len^2``
  boolean (16 MB at 4k context) out of every checkpoint.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from helm.hypercore.manifolds import Lorentz
from helm.hypercore.models.lorentz_feedforward import LorentzFeedForward
from helm.hypercore.nn.attention.lorentz_word_emb import LorentzEmbeddings
from helm.hypercore.nn.conv.conv_util_layers import LResNet, LorentzRMSNorm
from helm.hypercore.nn.linear.lorentz_linear import LorentzLinear

from .fused_ce import fused_linear_cross_entropy
from .hmla import LorentzKVCache, LorentzMLA
from .lorentz_ops import LorentzResidual
from .mice import LorentzMoE, LorentzSwiGLU
from .rope import precompute_freqs_cis, precompute_rope_cache

world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0

__all__ = ["Block", "LorentzDeepSeekV3", "HelmMiCE", "precompute_freqs_cis"]


class Block(nn.Module):
    """One HELM-MiCE layer: HMLA + (dense FFN | MiCE), both with Lorentz residuals."""

    def __init__(self, manifold: Lorentz, layer_id: int, args,
                 attn_impl: str = "flash", rope_impl: str = "complex",
                 fuse_experts: bool = True, fuse_residual: bool = True):
        super().__init__()
        self.manifold = manifold
        self.attn = LorentzMLA(manifold, args, attn_impl=attn_impl, rope_impl=rope_impl)
        if layer_id < args.n_dense_layers:
            self.ffn = (LorentzSwiGLU(manifold, args.dim, args.inter_dim) if fuse_experts
                        else LorentzFeedForward(manifold, args.dim, args.inter_dim))
        else:
            self.ffn = LorentzMoE(manifold, args, fuse_experts=fuse_experts,
                                  fuse_residual=fuse_residual)
        self.is_moe = isinstance(self.ffn, LorentzMoE)
        self.attn_norm = LorentzRMSNorm(manifold, args.dim - 1)
        self.ffn_norm = LorentzRMSNorm(manifold, args.dim - 1)
        residual = LorentzResidual if fuse_residual else LResNet
        self.attn_res = residual(manifold, use_scale=True,
                                 scale=math.sqrt(args.dim), learn_scale=False)
        self.ffn_res = residual(manifold, use_scale=True,
                                scale=math.sqrt(args.dim), learn_scale=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], is_causal: bool = False,
                cache: Optional[LorentzKVCache] = None):
        """Returns ``x``, or ``(x, indices, scores)`` when training a MiCE layer."""
        x = self.attn_res(x, self.attn(self.attn_norm(x), start_pos, freqs_cis, mask,
                                       is_causal=is_causal, cache=cache))
        if self.training and self.is_moe:
            x_ffn, idx, scores = self.ffn(self.ffn_norm(x))
            return self.ffn_res(x, x_ffn), idx, scores
        x_ffn = self.ffn(self.ffn_norm(x))
        if self.training:
            return self.ffn_res(x, x_ffn), None, None
        return self.ffn_res(x, x_ffn)


class HelmMiCE(nn.Module):
    """Hyperbolic decoder-only LM with latent attention and curvature experts.

    Args:
        args: model config (see ``config/args.py``).
        manifold_in / manifold_hidden / manifold_out: input, hidden and output
            Lorentz manifolds.
        attn_impl: ``"flash"`` (default) or ``"naive"``.
        rope_impl: ``"complex"`` (default) or ``"real"``; see
            :mod:`helm.modules.rope`.
        fuse_experts: fuse the SwiGLU gate/up projections into one GEMM.
        fuse_residual: use the fused Lorentz residual. Setting both ``fuse_*``
            to ``False`` with ``attn_impl="naive"`` and ``rope_impl="complex"``
            gives a configuration that is *bit-identical* to the reference.
        grad_checkpoint: recompute each block's activations in the backward pass.
    """

    def __init__(self, args, manifold_in, manifold_hidden, manifold_out,
                 attn_impl: str = "flash", rope_impl: str = "complex",
                 fuse_experts: bool = True, fuse_residual: bool = True,
                 grad_checkpoint: bool = False):
        super().__init__()
        global rank
        rank = dist.get_rank() if dist.is_initialized() else 0

        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.max_seq_len = args.max_seq_len
        self.train_curv = args.train_curv
        self.project_emb = args.project_emb
        self.rope_impl = rope_impl
        self.grad_checkpoint = grad_checkpoint

        if not self.project_emb:
            self.embed = LorentzEmbeddings(manifold_in, args.vocab_size, args.dim,
                                           manifold_out=manifold_hidden, posit_embed=False)
        else:
            self.embed = nn.Embedding(args.vocab_size, args.dim - 1)

        self.layers = nn.ModuleList([
            Block(manifold_hidden, i, args, attn_impl=attn_impl, rope_impl=rope_impl,
                  fuse_experts=fuse_experts, fuse_residual=fuse_residual)
            for i in range(args.n_layers)])
        self.final_proj = LorentzLinear(manifold_hidden, args.dim, args.dim - 1,
                                        manifold_out=manifold_out)
        self.norm = LorentzRMSNorm(manifold_out, args.dim - 1)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

        table = (precompute_rope_cache(args) if rope_impl == "real"
                 else precompute_freqs_cis(args))
        self.register_buffer("freqs_cis", table, persistent=False)
        # Non-persistent: a dense (max_seq_len, max_seq_len) bool is pure
        # derived state and would otherwise be written into every checkpoint.
        self.register_buffer(
            "attn_mask",
            torch.ones(args.max_seq_len, args.max_seq_len, dtype=torch.bool).triu(1),
            persistent=False)

    def _apply(self, fn, recurse: bool = True):
        """Move/cast the module while protecting the rotary table.

        The table is a block of constants, and ``nn.Module.to`` would happily
        rewrite it: casting to bf16 coarsens the rotation angles, and casting the
        *complex* layout to any real dtype throws away the imaginary part
        outright, silently degrading rotary embeddings to a cosine rescale. That
        is exactly what happens to the upstream model on ``.half()``,
        ``.bfloat16()`` or ``.to(dtype)`` -- it warns and carries on with a broken
        table. Restoring the original after the cast keeps device moves working
        while leaving the values alone.
        """
        rope = self.freqs_cis
        out = super()._apply(fn, recurse)
        self.freqs_cis = rope.to(device=self.freqs_cis.device)
        return out

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x_time = (x.square().sum(dim=-1, keepdim=True) + self.manifold_in.c).clamp_min(1e-12).sqrt()
        return torch.cat([x_time, x], dim=-1)

    def new_kv_caches(self, max_batch_size: int, max_seq_len: Optional[int] = None,
                      dtype: Optional[torch.dtype] = None,
                      device: Optional[torch.device] = None) -> List[LorentzKVCache]:
        """Allocate one :class:`LorentzKVCache` per layer, for incremental decoding."""
        max_seq_len = max_seq_len or self.max_seq_len
        device = device or self.head.weight.device
        dtype = dtype or self.head.weight.dtype
        return [LorentzKVCache(max_batch_size, max_seq_len, layer.attn.n_local_heads,
                               layer.attn.qk_head_dim, layer.attn.v_head_dim,
                               dtype=dtype, device=device)
                for layer in self.layers]

    def forward(self, tokens: torch.Tensor, start_pos: int = 0,
                attn_mask: Optional[torch.Tensor] = None,
                caches: Optional[List[LorentzKVCache]] = None,
                labels: Optional[torch.Tensor] = None,
                ce_chunk_size: int = 512):
        """
        Args:
            tokens: ``(B, N)`` token ids.
            start_pos: position of ``tokens[:, 0]`` in the full sequence.
            attn_mask: ``(B, N, N)`` boolean, ``True`` == *masked*. Typically the
                "different document" mask produced by sequence packing. When
                ``None``, a pure causal mask is applied without materialising it.
            caches: per-layer KV caches for incremental decoding.
            labels: when given, the cross-entropy loss is returned in place of
                the logits and computed without ever building them -- see
                :mod:`helm.modules.fused_ce`. Positions set to ``-100`` are
                skipped, including in the projection itself.
            ce_chunk_size: tokens per block in the fused head.

        Returns:
            ``logits`` (or ``loss`` if ``labels`` is given), plus
            ``(indices, scores)`` while training.
        """
        seqlen = tokens.size(-1)
        h = self.project(self.embed(tokens)) if self.project_emb else self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        # `is_causal=True` costs nothing and lets the flash kernel skip the whole
        # upper triangle; only fall back to an explicit mask when packing gives
        # us a document mask that a causal flag cannot express.
        is_causal = attn_mask is None and caches is None
        mask = None
        if attn_mask is not None:
            causal = self.attn_mask[start_pos:start_pos + seqlen, :start_pos + seqlen]
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)      # (B, N, N) -> (B, 1, N, N)
            mask = causal | attn_mask

        if self.training:
            all_indices: List[torch.Tensor] = []
            all_scores: List[torch.Tensor] = []
            for i, layer in enumerate(self.layers):
                cache = caches[i] if caches is not None else None
                if self.grad_checkpoint:
                    h, idx, scr = checkpoint(layer, h, start_pos, freqs_cis, mask,
                                             is_causal, cache, use_reentrant=False)
                else:
                    h, idx, scr = layer(h, start_pos, freqs_cis, mask, is_causal, cache)
                if idx is not None:
                    all_indices.append(idx)
                    all_scores.append(scr)
            h = self.norm(self.final_proj(h, return_space=True), space_only=True)
            return self._head(h, labels, ce_chunk_size), all_indices, all_scores

        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            h = layer(h, start_pos, freqs_cis, mask, is_causal, cache)
        h = self.norm(self.final_proj(h, return_space=True), space_only=True)
        return self._head(h, labels, ce_chunk_size)

    def _head(self, h, labels, ce_chunk_size):
        if labels is None:
            return self.head(h).float()
        return fused_linear_cross_entropy(h, self.head.weight, labels,
                                          bias=self.head.bias,
                                          chunk_size=ce_chunk_size)


# Upstream calls the model `LorentzDeepSeekV3`; keep the name importable so that
# existing scripts and checkpoints keep working.
LorentzDeepSeekV3 = HelmMiCE
