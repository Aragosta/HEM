#!/usr/bin/env python3
"""BrierLM -- a likelihood-free language-model metric, and samplers for both models.

CALM cannot report perplexity: it has no likelihood. It reports **BrierLM**
instead, an estimate of the Brier score built from *samples only*, which is
therefore computable for a discrete model as well. That is what makes it the one
metric on which a discrete HELM and a CALM-HELM can actually be compared -- see
`../ASSESSMENT.md` §3.1, where the absence of such a metric was flagged as the
thing that severs comparability with the HELM paper's benchmark table.

The estimator, from `upstream/models/modeling_calm.py::eval_brier`:

    brier_k = E[ 1{x1_{1..k} = y_{1..k}} + 1{x2_{1..k} = y_{1..k}}
                 - 1{x1_{1..k} = x2_{1..k}} ]

for two *independent* samples x1, x2 from the model and target y. The two
accuracy terms reward matching the target; the collision term penalises a model
that has collapsed onto a single confident guess, which is what makes it proper
rather than just an accuracy. BrierLM aggregates k = 1..4 as a geometric mean,
BLEU-style:

    BrierLM = (brier_1 · brier_2 · brier_3 · brier_4)^(1/4)

Usage: python CALM/experiments/brierlm.py   (runs the self-tests)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def brier_scores(samples_a: torch.Tensor, samples_b: torch.Tensor,
                 targets: torch.Tensor, max_n: int = 4) -> torch.Tensor:
    """Per-order Brier estimates from two independent sample sets.

    Args:
        samples_a, samples_b: ``(..., n)`` token ids, two *independent* draws.
        targets: ``(..., n)`` ground-truth ids.
        max_n: highest n-gram order; needs ``n >= max_n``.

    Returns:
        ``(max_n,)`` tensor of ``brier_1 ... brier_max_n``.
    """
    if samples_a.shape != samples_b.shape or samples_a.shape != targets.shape:
        raise ValueError(f"shape mismatch: {samples_a.shape}, {samples_b.shape}, "
                         f"{targets.shape}")
    if samples_a.size(-1) < max_n:
        raise ValueError(f"need at least {max_n} positions, got {samples_a.size(-1)}")

    # cumprod turns per-token equality into prefix equality: index k-1 is
    # "the whole length-k prefix matched".
    hit_a = torch.cumprod((samples_a[..., :max_n] == targets[..., :max_n]).float(), dim=-1)
    hit_b = torch.cumprod((samples_b[..., :max_n] == targets[..., :max_n]).float(), dim=-1)
    collision = torch.cumprod(
        (samples_a[..., :max_n] == samples_b[..., :max_n]).float(), dim=-1)

    estimate = hit_a + hit_b - collision
    return estimate.reshape(-1, max_n).mean(dim=0)


def brier_lm(samples_a: torch.Tensor, samples_b: torch.Tensor,
             targets: torch.Tensor, max_n: int = 4) -> float:
    """Geometric mean of the per-order Brier estimates (CALM reports this x100)."""
    scores = brier_scores(samples_a, samples_b, targets, max_n)
    product = scores.clamp_min(0).prod().item()
    return product ** (1.0 / max_n)


@torch.no_grad()
def sample_discrete(model, tokens: torch.Tensor, temperature: float = 1.0):
    """Two independent next-token sample sequences from a softmax model.

    Teacher-forced: position t is sampled from the model's distribution given the
    true prefix, which is the same conditioning the CALM sampler gets.
    """
    logits = model(tokens)
    logits = (logits[0] if isinstance(logits, tuple) else logits)[:, :-1].float()
    probs = torch.softmax(logits / temperature, dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    draw_a = torch.multinomial(flat, 1).reshape(probs.shape[:-1])
    draw_b = torch.multinomial(flat, 1).reshape(probs.shape[:-1])
    return draw_a, draw_b, tokens[:, 1:]


def self_test():
    """Anchor the estimator against cases whose value is known analytically."""
    torch.manual_seed(0)
    vocab, batch, length, max_n = 97, 64, 8, 4
    targets = torch.randint(0, vocab, (batch, length))

    perfect = brier_lm(targets, targets, targets, max_n)
    print(f"perfect model (always the target)      BrierLM {perfect:.4f}   expect 1.0")

    # Uniform model: brier_k = 2V^-k - V^-k = V^-k, so BrierLM = V^-2.5.
    # Checked at a *small* vocabulary on purpose. At V=97 a 4-gram collision has
    # probability 97^-4 ~ 1e-8, so brier_4 is empirically 0, the product is 0,
    # and any "assert got < expect" passes without testing anything.
    small_vocab, rows = 4, 200_000
    uniform_a = torch.randint(0, small_vocab, (rows, length))
    uniform_b = torch.randint(0, small_vocab, (rows, length))
    uniform_t = torch.randint(0, small_vocab, (rows, length))
    per_order = brier_scores(uniform_a, uniform_b, uniform_t, max_n)
    got = brier_lm(uniform_a, uniform_b, uniform_t, max_n)
    expect = small_vocab ** -2.5
    print(f"uniform model over {small_vocab} tokens          BrierLM {got:.5f}   "
          f"expect {expect:.5f}")
    for k in range(max_n):
        print(f"   brier_{k + 1}  {per_order[k]:.5f}   expect V^-{k + 1} = "
              f"{small_vocab ** -(k + 1):.5f}")

    # A model that always emits the same token, right or wrong: the accuracy
    # terms are small and the collision term is 1, so the score is driven
    # negative -- the collapse penalty doing its job.
    collapsed = torch.zeros_like(targets)
    scores = brier_scores(collapsed, collapsed, targets, max_n)
    print(f"collapsed model (one token always)     brier_1 {scores[0]:+.4f}   "
          f"expect negative")

    assert abs(perfect - 1.0) < 1e-9, perfect
    assert abs(got - expect) < 0.15 * expect, (got, expect)
    for k in range(max_n):
        predicted = small_vocab ** -(k + 1)
        assert abs(per_order[k].item() - predicted) < 0.15 * predicted, (k, per_order)
    assert scores[0] < 0, scores
    print("\nall self-tests passed")


if __name__ == "__main__":
    self_test()
