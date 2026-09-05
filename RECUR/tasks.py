"""Two task families, chosen so that one of them can move when the other cannot.

Ouro's most useful result is a *separation*: depth buys the ability to
manipulate facts, not the ability to store them. A study that measures only
language-model loss cannot see that separation, because next-token loss on
natural text is dominated by exactly the statistics depth is claimed not to
help with. So there are two families here and they are not interchangeable:

``bytes``
    WikiText-2 and PTB at byte level, official splits, bits per byte. This is
    the *storage-and-statistics* axis. The prediction from Ouro's mechanism is
    that recurrence buys little here. It is in the suite to keep that
    prediction falsifiable, and because a result on a real corpus is the only
    one comparable to anything published.

``hops`` / ``twochain``
    In-context composition. A random functional graph on ``n_entities`` is
    written into the context as pairs, and the model is asked for the
    ``h``-hop composition from a query entity. Every example carries a fresh
    graph, so nothing can be memorised into the weights: the task is pure
    manipulation, measured as accuracy on one answer token. ``twochain`` asks
    for a function of *two* chains, so a partial result has to survive while
    the second chain is computed -- that is the task the writable-state
    experiment needs, and it differs from ``hops`` in one field.

The hop count is the knob that makes the depth question falsifiable: if depth
buys sequential composition, the depth at which accuracy saturates should move
right as ``h`` grows, and should not move at all on ``bytes``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

DATA = Path(__file__).resolve().parents[1] / "CALM" / "data"
CORPORA = {
    "wikitext2": ("https://raw.githubusercontent.com/pytorch/examples/main/"
                  "word_language_model/data/wikitext-2", "wikitext2.{split}.txt"),
    "ptb": ("https://raw.githubusercontent.com/wojzaremba/lstm/master/data",
            "ptb.{split}.txt"),
}


# ------------------------------------------------------------------ byte LM

@dataclass
class ByteCorpus:
    name: str
    splits: Dict[str, torch.Tensor]
    digests: Dict[str, str]
    vocab_size: int = 256

    def describe(self) -> str:
        sizes = " / ".join(f"{k} {v.numel():,}" for k, v in self.splits.items())
        return f"{self.name} byte-level, official splits: {sizes}"


def load_bytes(name: str = "wikitext2") -> ByteCorpus:
    base, pattern = CORPORA[name]
    out, digests = {}, {}
    for split in ("train", "valid", "test"):
        path = DATA / pattern.format(split=split)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Fetch it rather than falling back to other "
                f"text:\n  curl -sSL -o {path} {base}/{split}.txt")
        raw = path.read_bytes()
        digests[split] = hashlib.sha256(raw).hexdigest()[:16]
        out[split] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    return ByteCorpus(name=name, splits=out, digests=digests)


def byte_batches(data: torch.Tensor, batch_size: int, seq_len: int, count: int,
                 seed: int) -> List[torch.Tensor]:
    """Fixed windows, so every arm sees a bit-identical data stream."""
    g = torch.Generator().manual_seed(seed)
    high = data.numel() - seq_len - 1
    return [torch.stack([data[i:i + seq_len + 1]
                         for i in torch.randint(0, high, (batch_size,), generator=g)])
            for _ in range(count)]


# --------------------------------------------------------- in-context hops

@dataclass
class HopSpec:
    """A composition task with **many supervised positions per example**.

    The first version of this task asked one question per sequence and could
    not be learned at any depth in the budget available: one supervised token
    per 36 is a thin gradient, and the pilot sat at ln(V) for 800 steps. The
    fix is not a bigger model, it is more questions -- ``queries`` of them per
    example, each with its own entity and hop count, all answered from the same
    context. Nothing else about the task changed, and the pilot then learned it.
    """

    n_entities: int = 16
    hops: Tuple[int, ...] = (1, 2, 3, 4)
    two_chain: bool = False
    queries: int = 8
    sorted_pairs: bool = False

    @property
    def max_hop(self) -> int:
        return max(self.hops)

    @property
    def vocab_size(self) -> int:
        # entities, then BOS, QUERY, and one token per hop count
        return self.n_entities + 2 + self.max_hop

    @property
    def query_width(self) -> int:
        return 5 if self.two_chain else 4      # QUERY HOP x [y] answer

    @property
    def seq_len(self) -> int:
        return 1 + 2 * self.n_entities + self.queries * self.query_width

    def answer_positions(self) -> List[int]:
        """Positions whose *next* token is an answer, i.e. where loss is taken."""
        base = 1 + 2 * self.n_entities
        return [base + m * self.query_width + self.query_width - 2
                for m in range(self.queries)]

    def describe(self) -> str:
        kind = "two-chain" if self.two_chain else "single-chain"
        return (f"{kind} in-context composition: {self.n_entities} entities, "
                f"hops {list(self.hops)}, {self.queries} queries/example, "
                f"vocab {self.vocab_size}, seq {self.seq_len}")


def _walk(perm: torch.Tensor, start: torch.Tensor, hops: torch.Tensor) -> torch.Tensor:
    """``f^h(start)`` per row, with a per-row hop count."""
    ans = start.clone()
    for i in range(int(hops.max())):
        step = torch.gather(perm, 1, ans)
        ans = torch.where(hops > i, step, ans)
    return ans


def hop_batch(spec: HopSpec, batch_size: int, generator: torch.Generator,
              hop: Optional[int] = None):
    """One batch: ``(tokens, targets, hops)`` with ``targets`` of shape (B, M).

    The graph is a random *permutation* of the entities, so every chain is
    infinite and no hop count is degenerate (a general functional graph lets
    short chains fall into fixed points, which would make a deep example
    accidentally shallow). Query entities within an example are distinct, so no
    question can be answered by copying another question's answer.
    """
    V, M = spec.n_entities, spec.queries
    BOS, QUERY, HOP0 = V, V + 1, V + 2

    perm = torch.stack([torch.randperm(V, generator=generator)
                        for _ in range(batch_size)])          # perm[b, s] = f(s)
    if spec.sorted_pairs:
        # Edges listed in entity order. The lookup is then available both by
        # content and by position, which is a strictly easier circuit to find;
        # whether it is *needed* is an empirical question about the budget, so
        # it is a flag rather than a decision baked into the task.
        order = torch.arange(V).unsqueeze(0).expand(batch_size, V).contiguous()
    else:
        order = torch.stack([torch.randperm(V, generator=generator)
                             for _ in range(batch_size)])
    pairs = torch.stack([order, torch.gather(perm, 1, order)],
                        dim=-1).reshape(batch_size, 2 * V)

    if hop is None:
        pick = torch.randint(0, len(spec.hops), (batch_size, M), generator=generator)
        hops = torch.tensor(spec.hops)[pick]
    else:
        hops = torch.full((batch_size, M), hop, dtype=torch.long)

    if M <= V:
        xs = torch.stack([torch.randperm(V, generator=generator)[:M]
                          for _ in range(batch_size)])
    else:                       # more questions than entities: repeats allowed
        xs = torch.randint(0, V, (batch_size, M), generator=generator)
    answers = _walk(perm, xs, hops)
    # Reject queries whose answer is the query entity itself. A random
    # permutation returns a fifth of entities to themselves within four hops,
    # and "copy the query" is a shortcut that would show up as depth doing
    # something. Resampling removes the shortcut rather than correcting for it
    # after the fact.
    for _ in range(8):
        bad = answers == xs
        if not bool(bad.any()):
            break
        fresh = torch.randint(0, V, xs.shape, generator=generator)
        xs = torch.where(bad, fresh, xs)
        answers = _walk(perm, xs, hops)
    if spec.two_chain:
        ys = (torch.stack([torch.randperm(V, generator=generator)[:M]
                           for _ in range(batch_size)]) if M <= V
              else torch.randint(0, V, (batch_size, M), generator=generator))
        answers = (answers + _walk(perm, ys, hops)) % V

    # The hop token comes *before* the query entity, so the position at which
    # the answer is predicted holds the entity itself. With the entity two
    # tokens back instead, the model has to move it forward before it can match
    # -- an extra composition step that a 4-block model at this budget did not
    # learn (see `RESULTS.md`, pilot 2). The task is otherwise unchanged.
    blocks = [torch.full((batch_size, M, 1), QUERY),
              (HOP0 + hops - 1).unsqueeze(-1), xs.unsqueeze(-1)]
    if spec.two_chain:
        blocks.append(ys.unsqueeze(-1))
    blocks += [answers.unsqueeze(-1)]
    tail = torch.cat(blocks, dim=-1).reshape(batch_size, M * spec.query_width)

    tokens = torch.cat([torch.full((batch_size, 1), BOS), pairs, tail], dim=1)
    return tokens, answers, hops


def hop_eval_set(spec: HopSpec, n_per_hop: int, seed: int):
    """A fixed evaluation set, identical for every arm, split by hop count."""
    g = torch.Generator().manual_seed(seed)
    return {h: hop_batch(spec, n_per_hop, g, hop=h) for h in spec.hops}
