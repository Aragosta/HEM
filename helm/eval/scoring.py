"""Batched scoring and generation for HELM models.

This is the machinery behind the ``lm-evaluation-harness`` integration, kept
separate so it is usable (and testable) without ``lm_eval`` installed.

The authors' evaluation wrapper scores **one continuation at a time**, and for
each one loops over its tokens in Python taking a fresh ``log_softmax`` over the
full 128256-entry vocabulary per token::

    for i in range(cont_len):
        log_probs = F.log_softmax(logits[0, prompt_len + i - 1], dim=-1)
        total_logprob += float(log_probs[cont_ids[0, i]])

That is one forward pass per request with ``batch_size`` ignored, plus a
host-synchronising ``float()`` per continuation token. Multiple-choice
benchmarks issue one request per answer option, so this is the dominant cost of
evaluating HELM.

Here the requests are length-sorted, batched, run through one forward pass per
batch, and reduced with a single vectorised gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ScoredContinuation:
    """Result of scoring one (context, continuation) pair."""

    logprob: float
    is_greedy: bool

    def as_tuple(self) -> Tuple[float, bool]:
        """The ``(logprob, is_greedy)`` pair lm-eval expects."""
        return self.logprob, self.is_greedy


def _forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the model in eval mode and return logits, whatever its arity."""
    out = model(input_ids)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def score_continuations(model, pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
                        *, batch_size: int = 8, max_seq_len: int = 2048,
                        pad_id: int = 0,
                        device: Optional[torch.device] = None,
                        progress: bool = False) -> List[ScoredContinuation]:
    """Score continuations under the model.

    Args:
        model: a HELM model in eval mode.
        pairs: ``(context_ids, continuation_ids)`` per request.
        batch_size: requests per forward pass.
        max_seq_len: context window; longer prompts are left-truncated, keeping
            the continuation intact.
        pad_id: token used to right-pad a batch. Padding is safe without a mask
            because attention is causal -- a padded position can only influence
            positions after it, and none of those are read.
        device: where to run; defaults to the model's device.
        progress: show a tqdm bar.

    Returns:
        One :class:`ScoredContinuation` per input pair, in the input order.
    """
    if device is None:
        device = next(model.parameters()).device

    prepared = []
    for index, (context_ids, cont_ids) in enumerate(pairs):
        cont = list(cont_ids)
        if not cont:
            raise ValueError(f"request {index} has an empty continuation")
        # Keep the whole continuation, drop context from the left.
        context = list(context_ids)[-(max_seq_len - len(cont)):]
        prepared.append((index, context, cont))

    # Length-sorted so each batch pads to roughly its own length rather than to
    # the longest request in the whole run.
    order = sorted(range(len(prepared)), key=lambda i: -(len(prepared[i][1])
                                                        + len(prepared[i][2])))
    results: List[Optional[ScoredContinuation]] = [None] * len(prepared)

    batches = range(0, len(order), batch_size)
    if progress:
        from tqdm import tqdm
        batches = tqdm(list(batches), desc="scoring", unit="batch")

    for start in batches:
        chunk = [prepared[i] for i in order[start:start + batch_size]]
        widths = [len(ctx) + len(cont) for _, ctx, cont in chunk]
        width = max(widths)

        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long,
                               device=device)
        for row, (_, ctx, cont) in enumerate(chunk):
            seq = torch.tensor(ctx + cont, dtype=torch.long, device=device)
            input_ids[row, :seq.numel()] = seq

        logits = _forward_logits(model, input_ids).float()
        log_probs = F.log_softmax(logits, dim=-1)
        greedy_ids = logits.argmax(dim=-1)

        for row, (index, ctx, cont) in enumerate(chunk):
            # Position t predicts token t+1, so continuation token j is
            # predicted from position len(ctx) + j - 1.
            first = len(ctx) - 1
            last = first + len(cont)
            targets = torch.tensor(cont, dtype=torch.long, device=device)
            token_lp = log_probs[row, first:last].gather(-1, targets[:, None]).squeeze(-1)
            results[index] = ScoredContinuation(
                # Summed in float64: the released loop accumulates into a Python
                # float, so a float32 reduction here would disagree with it by
                # ~5e-7 on a long continuation, purely from the accumulator.
                logprob=float(token_lp.double().sum()),
                is_greedy=bool((greedy_ids[row, first:last] == targets).all()),
            )

    return [r for r in results if r is not None]


@torch.no_grad()
def rolling_logprob(model, token_ids: Sequence[int], *, max_seq_len: int = 2048,
                    device: Optional[torch.device] = None) -> float:
    """Total log-probability of a sequence under the model.

    Windows the sequence when it exceeds the context length, scoring each token
    exactly once.

    The released implementation of this cannot run at all: it does
    ``out_logits, _, _ = self.model(inp)[:, -1]``, unpacking three values from a
    tensor slice, and feeds one token at a time with no cache, so even with the
    unpacking fixed every step would see a one-token context.
    """
    if device is None:
        device = next(model.parameters()).device
    tokens = list(token_ids)
    if len(tokens) < 2:
        return 0.0

    total = 0.0
    start = 0
    while start + 1 < len(tokens):
        window = tokens[start:start + max_seq_len]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        log_probs = F.log_softmax(_forward_logits(model, input_ids).float(), dim=-1)
        targets = torch.tensor(window[1:], dtype=torch.long, device=device)
        token_lp = log_probs[0, :len(window) - 1].gather(-1, targets[:, None])
        total += float(token_lp.double().sum())
        # Windows overlap by one token so the next window's first target is the
        # token after this window's last -- every token is scored exactly once.
        start += max_seq_len - 1
    return total


@torch.no_grad()
def generate(model, prompt_ids: Sequence[int], *, max_new_tokens: int = 64,
             temperature: float = 0.0, stop_ids: Sequence[int] = (),
             device: Optional[torch.device] = None) -> List[int]:
    """Greedy or sampled continuation, using the KV cache.

    The released wrapper raises ``NotImplementedError`` here ("Generation not
    required for MCQ evaluation"), which rules out every generative task in the
    harness. With the KV cache restored this is O(n) rather than O(n^2).

    Args:
        model: a HELM model exposing ``new_kv_caches`` (i.e. :class:`HelmMiCE`).
        prompt_ids: prompt token ids.
        max_new_tokens: how many tokens to produce.
        temperature: 0 for greedy, otherwise softmax sampling temperature.
        stop_ids: stop as soon as one of these is produced.
        device: defaults to the model's device.

    Returns:
        The generated token ids, excluding the prompt.
    """
    if not hasattr(model, "new_kv_caches"):
        raise TypeError(f"{type(model).__name__} has no KV cache; use HelmMiCE")
    if device is None:
        device = next(model.parameters()).device

    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    total = prompt.size(1) + max_new_tokens
    caches = model.new_kv_caches(1, max_seq_len=total, device=device)

    out = model(prompt, caches=caches)
    logits = out[0] if isinstance(out, tuple) else out

    produced: List[int] = []
    position = prompt.size(1)
    for _ in range(max_new_tokens):
        last = logits[:, -1].float()
        if temperature > 0:
            probs = torch.softmax(last / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = last.argmax(dim=-1, keepdim=True)
        token = int(next_id)
        produced.append(token)
        if token in stop_ids:
            break
        out = model(next_id, start_pos=position, caches=caches)
        logits = out[0] if isinstance(out, tuple) else out
        position += 1
    return produced
