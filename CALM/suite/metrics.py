"""Metrics, each labelled with the cells it exists in and the comparisons it licenses.

The recurring failure this module guards against is not a wrong formula. It is
reporting a number in a cell where it is undefined, or comparing two numbers
that are not on the same scale, or quoting an aggregate that reads the same for
every model. Each function below carries the scope it is valid over.
"""

from __future__ import annotations

import collections
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from brierlm import brier_scores  # noqa: E402


# ------------------------------------------------------- quality, all cells

@torch.no_grad()
def top1_accuracy(draw: Callable, batches: Sequence[torch.Tensor],
                  n_samples: int = 32) -> float:
    """Modal next-byte accuracy. **The one metric on the same scale in all four cells.**

    ``draw(tokens, n)`` returns ``(samples, targets)`` with samples
    ``(n, B, S')``. For a discrete model the draws come from its softmax; for a
    CALM model from its head, decoded through the frozen autoencoder. Both end
    as byte ids, which is what makes the comparison legitimate.
    """
    correct = total = 0
    for tokens in batches:
        samples, targets = draw(tokens, n_samples)
        predicted = torch.mode(samples, dim=0).values
        correct += (predicted == targets).sum().item()
        total += targets.numel()
    return correct / total


@torch.no_grad()
def brier_by_order(draw: Callable, batches: Sequence[torch.Tensor],
                   max_n: int = 4) -> List[float]:
    """BrierLM per n-gram order. Valid in all four cells.

    Returned per order and never only as the geometric mean: at byte level two
    independent draws agree on a 4-gram with probability around 256^-4, so
    ``brier_4`` pins at 0 and the aggregate reads 0.0000 for a strong model and
    a random one alike. Reading:

    ``~0``   diffuse and wrong -- draws match neither the target nor each other
    ``< 0``  mode collapse -- the draws agree with each other but not the target
    ``> 0``  real signal, and the only region where comparing cells means anything
    """
    totals = torch.zeros(max_n, dtype=torch.float64)
    for tokens in batches:
        first, targets = draw(tokens, 1)
        second, _ = draw(tokens, 1)
        totals += brier_scores(first[0], second[0], targets, max_n).double()
    return (totals / len(batches)).tolist()


@torch.no_grad()
def bits_per_byte(logits_fn: Callable, batches: Sequence[torch.Tensor]) -> float:
    """Held-out ``-log2 p(byte)``. **Discrete cells only.**

    There is no CALM equivalent and none is coming: an implicit sampler has no
    density. That is a property of CALM's design, not a gap in this suite, and
    the runner records ``None`` rather than substituting a proxy.
    """
    nats = count = 0.0
    for tokens in batches:
        logits = logits_fn(tokens[:, :-1])
        target = tokens[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), target.reshape(-1))
        nats += loss.item() * target.numel()
        count += target.numel()
    return nats / count / math.log(2)


# ------------------------------------------------------------- the floor

def lookup_baselines(train: torch.Tensor, batches: Sequence[torch.Tensor],
                     vocab: int = 256) -> Dict[str, float]:
    """What counting achieves on the same held-out bytes.

    Non-negotiable. A previous run in this project reported 9.23% held-out
    accuracy as a model comparison when a bigram lookup table scored 19.86% on
    the same split; without this row that was invisible.
    """
    counts = collections.Counter(train.tolist())
    mode = counts.most_common(1)[0][0]
    following = collections.defaultdict(collections.Counter)
    for a, b in zip(train[:-1].tolist(), train[1:].tolist()):
        following[a][b] += 1
    best = {k: v.most_common(1)[0][0] for k, v in following.items()}
    totals = {k: sum(v.values()) for k, v in following.items()}

    correct = seen = 0
    nats = 0.0
    for tokens in batches:
        for row in tokens:
            ids = row.tolist()
            for a, b in zip(ids[:-1], ids[1:]):
                seen += 1
                correct += best.get(a, mode) == b
                table = following.get(a)
                nats -= math.log(((table.get(b, 0) if table else 0) + 1)
                                 / (totals.get(a, 0) + vocab))
    held = torch.cat([t.reshape(-1) for t in batches])
    return {"bigram_top1": correct / seen,
            "bigram_bpb": nats / seen / math.log(2),
            "unigram_top1": (held == mode).float().mean().item(),
            "uniform_top1": 1.0 / vocab,
            "uniform_bpb": math.log2(vocab)}


# --------------------------------------------------- training dynamics

def steps_to_threshold(history: Sequence[float], threshold: float) -> int:
    """First step whose running mean of 20 falls below ``threshold``; -1 if never.

    Convergence speed, not just final loss. Two cells can land in the same place
    having taken very different paths, and under an objective as
    high-variance as the energy score that difference is worth seeing.
    """
    window: List[float] = []
    for step, value in enumerate(history):
        window.append(value)
        if len(window) > 20:
            window.pop(0)
        if len(window) == 20 and sum(window) / 20 < threshold:
            return step
    return -1


@torch.no_grad()
def effective_rank(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Participation ratio of the covariance spectrum: ``(sum l)^2 / sum l^2``.

    A cheap collapse detector. A model can have a falling loss and a rank-3
    representation, and accuracy alone will not show it. Reported next to
    accuracy so a "good" cell using three directions is visible as such.
    """
    x = activations.reshape(-1, activations.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    if x.shape[0] < 2:
        return 0.0
    eigenvalues = torch.linalg.svdvals(x).pow(2)
    total = eigenvalues.sum()
    if total <= eps:
        return 0.0
    return (total.pow(2) / eigenvalues.pow(2).sum().clamp_min(eps)).item()


def gradient_percentiles(norms: Sequence[float]) -> Dict[str, float]:
    """Median and tail of the gradient-norm distribution.

    The energy score is a high-variance objective, and the wrapped-normal latent
    turned out to depend on gradient clipping for whether it scored 2.29% or
    80.91%. The tail is the part that decides that, so it is recorded rather
    than assumed benign.
    """
    if not norms:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(norms)

    def at(q: float) -> float:
        return ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)]

    return {"p50": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": ordered[-1]}


# ------------------------------------------------------------- efficiency

def efficiency(patch_size: int, seq_len: int, seconds_per_step: float,
               tokens_per_step: int, peak_bytes: int) -> Dict[str, float]:
    """The tradeoff CALM claims, measured rather than asserted.

    ``ar_steps_per_token`` is ``1 / patch_size``: the structural claim, true by
    construction. The wall-clock and memory figures are what decide whether it
    converts into an actual saving.
    """
    return {"ar_steps_per_token": 1.0 / patch_size,
            "ar_steps_per_sequence": seq_len / patch_size,
            "seconds_per_token": seconds_per_step / max(tokens_per_step, 1),
            "peak_mb": peak_bytes / (1024 ** 2)}
