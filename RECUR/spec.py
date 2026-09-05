"""The one place the scale of the study is decided.

Every experiment imports its task shape, model shape and training budget from
here, so that "the same model" means the same model across E0-E5 and a change
of scale is one edit rather than six. The values are not arbitrary; each is the
outcome of a pilot recorded in `RESULTS.md`:

``hop_spec(h)``
    6 entities, one hop count per run, **6 questions per context**. One
    question per context gave one supervised token in 36 and did not learn at
    any depth; twelve entities with hops mixed 1-4 did not learn either.
    Queries whose answer is the query entity are rejected, because a random
    permutation returns a fifth of entities to themselves within four hops and
    "copy the query" would otherwise look like depth working.

``HOPS_CFG``
    dim 64, 4 heads, prelude 1 / core 2 / coda 1. Four blocks at R=1 and ten at
    R=4. Small enough that four of them fit on four cores; large enough that
    the composition circuit exists.

``STEPS``
    The budget at which the pilot's baseline is past the induction phase
    transition and still improving. Below it every arm sits at chance and the
    experiment measures nothing; the pilot log in `RESULTS.md` is the evidence.

Byte-level runs use their own width because their vocabulary is 256 rather
than 18, and a 64-wide model spends most of its parameters in the embedding.
"""

from __future__ import annotations

N_ENTITIES = 6
QUERIES = 6


def hop_spec(hop: int = 2, two_chain: bool = False) -> dict:
    """The task at one fixed hop count.

    Hop count is fixed *per run* rather than mixed within a batch. Mixing four
    hop counts means each circuit sees a quarter of the gradient and, measured
    in the pilot, none of them formed inside the budget. Fixing it per run also
    makes the depth question sharper: hop count becomes an experimental factor
    with its own axis rather than a within-batch nuisance.
    """
    return {"n_entities": N_ENTITIES, "hops": (hop,), "queries": QUERIES,
            "sorted_pairs": False}


HOP_SPEC = hop_spec(2)

HOPS_CFG = {"dim": 64, "n_heads": 4, "n_prelude": 1, "n_core": 2, "n_coda": 1}
BYTE_CFG = {"dim": 128, "n_heads": 4, "n_prelude": 1, "n_core": 2, "n_coda": 1}

STEPS = 3000                 # composition tasks (AttnRes needs ~1.6x
                             # the steps of standard residuals; see E0)
BYTE_STEPS = 600             # byte-level LM
BATCH = 32
LR = 3e-3
BYTE_BATCH = 16
BYTE_SEQ = 128
BYTE_LR = 1.5e-3
