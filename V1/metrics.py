"""BrierLM — the one metric both models can be scored on.

CALM-K3 has no likelihood, so it has no perplexity. Comparing it to a softmax
baseline on the baseline's own metric is impossible, and comparing training
losses is meaningless: one is a cross-entropy in nats, the other an energy
distance in latent space. They are not the same quantity and neither bounds
the other.

BrierLM is the CALM paper's answer. It estimates the Brier score — a strictly
proper scoring rule — from *samples only*:

    brier_k = E[ 1{a₁..ₖ = y₁..ₖ} + 1{b₁..ₖ = y₁..ₖ} − 1{a₁..ₖ = b₁..ₖ} ]

for two independent draws `a`, `b` and target `y`. The two accuracy terms
reward matching the target; the collision term punishes a model that has
collapsed onto one confident guess, and it is what makes this proper rather
than a dressed-up accuracy — a model that always emits the same token scores
*negative*. Orders 1..4 are combined BLEU-style as a geometric mean:

    BrierLM = (brier₁ · brier₂ · brier₃ · brier₄)^(1/4)

Because it only ever looks at sampled ids, a discrete model and a continuous
one can be put on the same axis. `self_test()` anchors the estimator on two
cases with known values: a perfect model scores 1, and a uniform model over a
vocabulary of `V` scores `V^-2.5`.
"""

from __future__ import annotations

import torch


def brier_scores(
    samples_a: torch.Tensor,
    samples_b: torch.Tensor,
    targets: torch.Tensor,
    max_n: int = 4,
) -> torch.Tensor:
    """Per-order Brier estimates from two independent sample sets.

    Args:
        samples_a, samples_b: `(..., n)` token ids, two *independent* draws.
        targets: `(..., n)` ground-truth ids.
        max_n: highest n-gram order; needs `n >= max_n`.

    Returns:
        `(max_n,)` — `brier_1 ... brier_max_n`.
    """
    if samples_a.shape != samples_b.shape or samples_a.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: {tuple(samples_a.shape)}, "
            f"{tuple(samples_b.shape)}, {tuple(targets.shape)}"
        )
    if samples_a.size(-1) < max_n:
        raise ValueError(
            f"need at least {max_n} positions, got {samples_a.size(-1)}"
        )

    # Every non-overlapping window of `max_n` positions is one observation.
    # Scoring only the first `max_n` columns — which is what a naive reading
    # of the estimator does — throws away all but the opening n-gram of each
    # row and leaves the 3- and 4-gram orders with almost no sample.
    def windows(tensor: torch.Tensor) -> torch.Tensor:
        usable = tensor.size(-1) - tensor.size(-1) % max_n
        return tensor[..., :usable].reshape(-1, max_n)

    windowed_a, windowed_b, windowed_t = (
        windows(samples_a), windows(samples_b), windows(targets)
    )

    # cumprod turns per-token equality into prefix equality: index k-1 is
    # "the whole length-k prefix matched".
    hit_a = torch.cumprod((windowed_a == windowed_t).float(), dim=-1)
    hit_b = torch.cumprod((windowed_b == windowed_t).float(), dim=-1)
    collision = torch.cumprod((windowed_a == windowed_b).float(), dim=-1)
    return (hit_a + hit_b - collision).mean(dim=0)


def brier_lm(
    samples_a: torch.Tensor,
    samples_b: torch.Tensor,
    targets: torch.Tensor,
    max_n: int = 4,
) -> float:
    """Geometric mean of the per-order estimates. CALM reports this x100."""
    scores = brier_scores(samples_a, samples_b, targets, max_n)
    return scores.clamp_min(0).prod().item() ** (1.0 / max_n)


@torch.no_grad()
def sample_pairs_calm(
    model,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two independent teacher-forced continuations from CALM-K3.

    Returns `(a, b, targets)` flattened back to token order, so the n-gram
    prefixes BrierLM measures run across patch boundaries exactly as they do
    for the baseline.
    """
    draw_a = model.predict_tokens(
        input_ids, generator=generator, temperature=temperature
    )
    draw_b = model.predict_tokens(
        input_ids, generator=generator, temperature=temperature
    )
    batch = input_ids.size(0)
    targets = input_ids[:, model.patch_size :]
    return (
        draw_a.reshape(batch, -1),
        draw_b.reshape(batch, -1),
        targets.reshape(batch, -1),
    )


@torch.no_grad()
def sample_pairs_baseline(
    model,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two independent teacher-forced next-token draws from a softmax model."""
    logits = model(input_ids=input_ids).logits[:, :-1].float()
    probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    flat = probabilities.reshape(-1, probabilities.size(-1))
    draw_a = torch.multinomial(flat, 1, generator=generator).reshape(
        probabilities.shape[:-1]
    )
    draw_b = torch.multinomial(flat, 1, generator=generator).reshape(
        probabilities.shape[:-1]
    )
    return draw_a, draw_b, input_ids[:, 1:]


def self_test() -> None:
    """Anchor the estimator on cases whose value is known analytically."""
    torch.manual_seed(0)
    length, max_n = 8, 4
    targets = torch.randint(0, 97, (64, length))

    perfect = brier_lm(targets, targets, targets, max_n)
    assert abs(perfect - 1.0) < 1e-9, perfect

    # Uniform over V: brier_k = 2V^-k - V^-k = V^-k, so BrierLM = V^-2.5.
    # Deliberately at a small V: at V=97 a 4-gram collision has probability
    # 1e-8, brier_4 is empirically 0, and the test would pass vacuously.
    vocab, rows = 4, 200_000
    a = torch.randint(0, vocab, (rows, length))
    b = torch.randint(0, vocab, (rows, length))
    t = torch.randint(0, vocab, (rows, length))
    per_order = brier_scores(a, b, t, max_n)
    got, expect = brier_lm(a, b, t, max_n), vocab**-2.5
    assert abs(got - expect) < 0.15 * expect, (got, expect)
    for k in range(max_n):
        predicted = vocab ** -(k + 1)
        assert abs(per_order[k].item() - predicted) < 0.15 * predicted, (k, per_order)

    # Collapsed model: accuracy terms small, collision term 1 -> negative.
    collapsed = torch.zeros_like(targets)
    assert brier_scores(collapsed, collapsed, targets, max_n)[0] < 0

    print("BrierLM self-tests passed "
          f"(perfect {perfect:.4f}, uniform {got:.5f} vs expected {expect:.5f})")


__all__ = [
    "brier_scores",
    "brier_lm",
    "sample_pairs_calm",
    "sample_pairs_baseline",
    "self_test",
]


if __name__ == "__main__":
    self_test()
