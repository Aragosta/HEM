"""A small synthetic corpus, generated locally and deterministically.

The repository's own text loaders reach for HuggingFace, which is not
available in this environment, and a comparison run on random tokens measures
nothing: with an i.i.d. uniform corpus every model's best strategy is the
uniform distribution and both sides score the same by construction.

So the corpus here is *learnable but not trivial*: a sparse first-order Markov
chain (each token has a handful of plausible successors) with motifs — fixed
token n-grams — injected at intervals. That gives structure at two scales:
local transitions a token-level model picks up immediately, and repeated
blocks that survive being compressed into a patch. Both models see exactly the
same stream from the same seed.

`markov_entropy_floor` reports the corpus's own conditional entropy, so a
model's cross-entropy can be read against the best achievable value rather
than against zero.
"""

from __future__ import annotations

import math

import torch


class SyntheticCorpus:
    """A sparse Markov chain over `vocab_size` tokens with repeated motifs."""

    def __init__(
        self,
        vocab_size: int = 64,
        successors: int = 4,
        num_motifs: int = 8,
        motif_length: int = 6,
        motif_probability: float = 0.15,
        seed: int = 0,
    ):
        if successors >= vocab_size:
            raise ValueError("successors must be smaller than vocab_size")
        self.vocab_size = vocab_size
        self.successors = successors
        self.motif_probability = motif_probability
        generator = torch.Generator().manual_seed(seed)

        # Each token allows `successors` continuations with a skewed
        # distribution: uniform successors would make the floor flat and the
        # ordering trivial to learn.
        self.next_tokens = torch.stack(
            [
                torch.randperm(vocab_size, generator=generator)[:successors]
                for _ in range(vocab_size)
            ]
        )
        weights = torch.rand(vocab_size, successors, generator=generator) + 0.1
        self.next_probs = weights / weights.sum(dim=-1, keepdim=True)

        self.motifs = torch.randint(
            0, vocab_size, (num_motifs, motif_length), generator=generator
        )

    def markov_entropy_floor(self) -> float:
        """Conditional entropy of the chain in nats, ignoring motifs.

        A model cannot beat this on the Markov part of the stream; motifs are
        more predictable still, so the true floor is a little lower.
        """
        probabilities = self.next_probs
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        return entropy.mean().item()

    def sample(self, batch: int, seq_len: int, seed: int = 0) -> torch.Tensor:
        """`(batch, seq_len)` token ids."""
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.zeros(batch, seq_len, dtype=torch.long)
        tokens[:, 0] = torch.randint(
            0, self.vocab_size, (batch,), generator=generator
        )

        position = 1
        motif_countdown = torch.zeros(batch, dtype=torch.long)
        motif_choice = torch.zeros(batch, dtype=torch.long)
        while position < seq_len:
            previous = tokens[:, position - 1]

            # Continue any motif already in flight.
            active = motif_countdown > 0
            step = torch.zeros(batch, dtype=torch.long)
            if active.any():
                index = self.motifs.size(1) - motif_countdown
                step[active] = self.motifs[motif_choice[active], index[active]]
                motif_countdown[active] -= 1

            # Otherwise draw a Markov successor, and occasionally start a motif.
            inactive = ~active
            if inactive.any():
                rows = self.next_probs[previous[inactive]]
                picked = torch.multinomial(rows, 1, generator=generator).squeeze(-1)
                step[inactive] = self.next_tokens[previous[inactive], picked]
                starts = (
                    torch.rand(int(inactive.sum()), generator=generator)
                    < self.motif_probability
                )
                if starts.any():
                    indices = inactive.nonzero(as_tuple=True)[0][starts]
                    motif_choice[indices] = torch.randint(
                        0, self.motifs.size(0), (len(indices),), generator=generator
                    )
                    motif_countdown[indices] = self.motifs.size(1)

            tokens[:, position] = step
            position += 1
        return tokens

    def batches(
        self, num_batches: int, batch: int, seq_len: int, seed: int = 0
    ):
        """Deterministic stream of distinct batches."""
        for index in range(num_batches):
            yield self.sample(batch, seq_len, seed=seed + index)


def perplexity_floor(corpus: SyntheticCorpus) -> float:
    return math.exp(corpus.markov_entropy_floor())


__all__ = ["SyntheticCorpus", "perplexity_floor"]
