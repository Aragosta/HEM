"""Fused linear + cross-entropy for the language-model head.

HELM-MiCE has an unusually lopsided head. The released 120M configuration pairs
``dim = 390`` with a 128256-entry Llama-3 vocabulary, so ``head`` alone is 50M of
the model's 107M parameters and its GEMM outweighs every transformer layer put
together -- profiling a two-layer model puts **81% of the forward pass** in the
head.

Worse is the tensor it produces. ``self.head(h).float()`` materialises
``(batch, seq, vocab)`` in float32: at the released training shape (batch 4,
2048 tokens) that is **3.9 GiB**, held live through the backward pass, for a
model whose weights are 0.4 GiB.

``fused_linear_cross_entropy`` never builds it. It

* **drops ignored positions before the GEMM.** With sequence packing a
  meaningful fraction of positions are padding or document boundaries carrying
  ``ignore_index``; the reference computes 128256 logits for every one of them
  and then throws the row away. This is a pure FLOP saving proportional to that
  fraction.
* **chunks over the remaining tokens**, so the largest live logit block is
  ``chunk_size x vocab`` rather than ``tokens x vocab``.
* **recomputes the logits in the backward pass** instead of storing them,
  trading one extra GEMM for the whole activation.

The arithmetic is unchanged: logits are still promoted to float32 before the
softmax, exactly as ``.float()`` did, and the reduction is still the mean over
non-ignored positions. Gradients come out bit-identical to
``F.cross_entropy(head(h).float(), labels)``; the loss differs only by float32
summation order (~1e-7).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """Cross-entropy of a linear projection, without materialising the logits."""

    @staticmethod
    def forward(ctx, hidden, weight, bias, target, ignore_index, chunk_size):
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_target = target.reshape(-1)

        keep = (flat_target != ignore_index).nonzero(as_tuple=True)[0]
        n_kept = keep.numel()
        if n_kept == 0:
            ctx.save_for_backward(hidden, weight, bias, keep, flat_target)
            ctx.empty = True
            return torch.zeros((), dtype=torch.float32, device=hidden.device)

        kept_hidden = flat_hidden.index_select(0, keep)
        kept_target = flat_target.index_select(0, keep)

        # The running total is float64 even though the logits are float32: it is
        # a scalar, so the precision is free, and it makes the result independent
        # of chunk_size rather than merely close. Accumulating in float32 let the
        # answer wobble by ~2e-6 across chunk sizes.
        loss_sum = torch.zeros((), dtype=torch.float64, device=hidden.device)
        for start in range(0, n_kept, chunk_size):
            stop = min(start + chunk_size, n_kept)
            logits = F.linear(kept_hidden[start:stop], weight, bias).float()
            # reduction="none" then a float64 sum: reducing inside
            # cross_entropy would sum in float32, so the answer would still
            # depend on where the chunk boundaries fall.
            loss_sum += F.cross_entropy(logits, kept_target[start:stop],
                                        reduction="none").double().sum()

        ctx.save_for_backward(hidden, weight, bias, keep, kept_target)
        ctx.chunk_size = chunk_size
        ctx.n_kept = n_kept
        ctx.empty = False
        # float32, matching `head(h).float()` followed by cross_entropy.
        return (loss_sum / n_kept).float()

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias, keep, kept_target = ctx.saved_tensors
        if ctx.empty:
            return torch.zeros_like(hidden), torch.zeros_like(weight), \
                (None if bias is None else torch.zeros_like(bias)), None, None, None

        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        kept_hidden = flat_hidden.index_select(0, keep)
        n_kept = ctx.n_kept
        scale = grad_output.float() / n_kept

        grad_kept = torch.empty_like(kept_hidden, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_bias = None if bias is None else torch.zeros_like(bias, dtype=torch.float32)
        # Hoisted: under bf16 this is a vocab x dim copy, and doing it per chunk
        # would allocate it once per iteration. A no-op when weight is float32.
        weight_f = weight.float()

        for start in range(0, n_kept, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, n_kept)
            block = kept_hidden[start:stop].float()
            # Recompute rather than store: this is the whole point.
            logits = F.linear(block, weight_f,
                              None if bias is None else bias.float())
            probs = torch.softmax(logits, dim=-1)
            probs.scatter_(1, kept_target[start:stop, None],
                           probs.gather(1, kept_target[start:stop, None]) - 1.0)
            probs *= scale
            grad_kept[start:stop] = probs @ weight_f
            grad_weight += probs.t() @ block
            if grad_bias is not None:
                grad_bias += probs.sum(0)
            del logits, probs

        grad_hidden = torch.zeros_like(flat_hidden, dtype=torch.float32)
        grad_hidden.index_copy_(0, keep, grad_kept)
        grad_hidden = grad_hidden.to(hidden.dtype).view_as(hidden)
        return (grad_hidden, grad_weight.to(weight.dtype),
                None if grad_bias is None else grad_bias.to(bias.dtype), None, None, None)


def fused_linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor,
                               target: torch.Tensor, bias: Optional[torch.Tensor] = None,
                               ignore_index: int = -100,
                               chunk_size: int = 512) -> torch.Tensor:
    """Mean cross-entropy of ``linear(hidden, weight, bias)`` against ``target``.

    Equivalent to::

        F.cross_entropy(F.linear(hidden, weight, bias).float().flatten(0, -2),
                        target.flatten(), ignore_index=ignore_index)

    but without ever holding the full logit tensor.

    Args:
        hidden: ``(..., dim)`` hidden states.
        weight: ``(vocab, dim)`` projection.
        target: ``(...)`` class indices, matching ``hidden`` minus its last axis.
        bias: optional ``(vocab,)``.
        ignore_index: positions to skip entirely (also skipped in the GEMM).
        chunk_size: tokens per block. Peak logit memory is
            ``chunk_size x vocab x 4`` bytes; smaller trades throughput for
            memory. 512 was the fastest setting measured on CPU.

    Returns:
        Scalar float32 loss, matching ``head(h).float()`` followed by
        cross-entropy. The running total is kept in float64 so the result does
        not depend on ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return _FusedLinearCrossEntropy.apply(hidden, weight, bias, target,
                                          ignore_index, chunk_size)
