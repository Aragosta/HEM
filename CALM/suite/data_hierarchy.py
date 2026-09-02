#!/usr/bin/env python3
"""Is there a hierarchy for patches to have? Measured in the data, no model involved.

HELM's premise is that language has hierarchical, tree-like structure that
hyperbolic space embeds with low distortion. The premise is about **tokens**:
word co-occurrence graphs are scale-free, rare words sit at the leaves.

CALM replaces token prediction with patch prediction, which moves the premise:
the question becomes whether **K-token patches** carry the same structure. Every
experiment in this repository has tried to answer that through a trained model,
where the answer is confounded with capacity, optimisation and our own bugs.

This asks the data directly. Build a co-occurrence graph over the most frequent
units, turn it into a metric space, and measure Gromov's delta -- the standard
justification for hyperbolic embeddings in the first place. Do it once for
tokens and once for K-grams, on the same corpus, with the same construction.

* ``delta_patch > delta_token`` -- patches are **less** tree-like than the tokens
  they are built from. Patching flattens the structure HELM exploits, and a
  hyperbolic backbone has less to work with under CALM's objective. This is the
  ``HIERARCHY.md`` worry, answered in the data rather than through a model.
* ``delta_patch ~ delta_token`` -- the structure survives aggregation, and a
  hyperbolic HELM-CALM has the same premise available that HELM does.
* ``delta_patch < delta_token`` -- patching *concentrates* hierarchy. Would be
  the strongest case for HELM-CALM and the one to check hardest.

A shuffled control is included. Any construction like this produces *some*
delta; the number only means something against the delta of the same
construction on text with the word order destroyed.

Usage: python CALM/suite/data_hierarchy.py --corpus wikitext2 --units 400
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "suite"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from corpus import load  # noqa: E402
from probes import delta_hyperbolicity  # noqa: E402


def ngrams(sequence: Sequence[int], k: int) -> List[Tuple[int, ...]]:
    """Non-overlapping K-grams -- the patches CALM would actually form."""
    return [tuple(sequence[i:i + k]) for i in range(0, len(sequence) - k + 1, k)]


def cooccurrence_distances(units: Sequence, top: int, window: int
                           ) -> Tuple[torch.Tensor, List]:
    """A metric space from co-occurrence, via positive PMI.

    Distance is ``1 / (1 + PPMI)``: units that co-occur far more than chance are
    close, unrelated ones approach 1. PPMI rather than raw counts so a unit is
    not close to everything simply by being frequent -- which would manufacture a
    hub-and-spoke shape and hand back a low delta for free.
    """
    counts = collections.Counter(units)
    vocabulary = [unit for unit, _ in counts.most_common(top)]
    index = {unit: i for i, unit in enumerate(vocabulary)}
    n = len(vocabulary)

    joint = torch.zeros((n, n), dtype=torch.float64)
    for position, unit in enumerate(units):
        i = index.get(unit)
        if i is None:
            continue
        for offset in range(1, window + 1):
            if position + offset >= len(units):
                break
            j = index.get(units[position + offset])
            if j is not None:
                joint[i, j] += 1
                joint[j, i] += 1

    total = joint.sum().clamp_min(1)
    marginal = joint.sum(1, keepdim=True).clamp_min(1)
    expected = (marginal @ marginal.T) / total
    ppmi = (joint * total / expected.clamp_min(1e-12)).clamp_min(1e-12).log().clamp_min(0)
    distances = 1.0 / (1.0 + ppmi)
    distances.fill_diagonal_(0)
    return distances, vocabulary


def measure(units: Sequence, label: str, top: int, window: int, seed: int) -> Dict:
    distances, vocabulary = cooccurrence_distances(units, top, window)
    delta = delta_hyperbolicity(distances, samples=200000, seed=seed)
    density = (distances < 0.999).float().mean().item()
    return {"label": label, "units": len(units), "vocabulary": len(vocabulary),
            "delta": delta, "linked_fraction": density}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wikitext2")
    parser.add_argument("--level", default="word", choices=("word", "byte"))
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--units", type=int, default=400,
                        help="most frequent units to build the graph over")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()

    corpus = load(options.corpus, level=options.level)
    sequence = corpus.train[:options.limit].tolist()
    print(f"{corpus.name} {options.level}-level, {len(sequence):,} units, "
          f"graph over the {options.units} most frequent, window {options.window}\n")

    rows = [measure(sequence, f"tokens (K=1)", options.units, options.window,
                    options.seed)]
    for k in (2, options.patch, 8):
        rows.append(measure(ngrams(sequence, k), f"patches (K={k})",
                            options.units, options.window, options.seed))

    generator = torch.Generator().manual_seed(options.seed)
    shuffled = torch.tensor(sequence)[torch.randperm(len(sequence),
                                                     generator=generator)].tolist()
    rows.append(measure(shuffled, "tokens, order shuffled", options.units,
                        options.window, options.seed))

    print(f"{'construction':>26s} {'delta':>8s} {'linked':>8s} {'vocab':>7s}")
    for row in rows:
        print(f"{row['label']:>26s} {row['delta']:8.4f} "
              f"{row['linked_fraction']:8.1%} {row['vocabulary']:7d}")

    token_delta = rows[0]["delta"]
    patch_delta = next(r["delta"] for r in rows
                       if r["label"] == f"patches (K={options.patch})")
    shuffled_delta = rows[-1]["delta"]
    print(f"\nlower delta = more tree-like. Read against the shuffled control "
          f"({shuffled_delta:.4f}):\nany construction yields some delta, so only "
          f"the gap to shuffled text is structure.")
    print(f"\ntoken delta {token_delta:.4f}  ->  K={options.patch} patch delta "
          f"{patch_delta:.4f}   ratio {patch_delta / token_delta:.3f}")
    if patch_delta > token_delta * 1.05:
        print("  Patches are LESS tree-like than their tokens. Patching flattens "
              "the structure\n  HELM exploits -- the HIERARCHY.md worry, in the "
              "data rather than through a model.")
    elif patch_delta < token_delta * 0.95:
        print("  Patches are MORE tree-like. The strongest case for HELM-CALM, "
              "and the one to\n  check hardest before believing.")
    else:
        print("  The structure survives aggregation: a hyperbolic HELM-CALM has "
              "the same premise\n  available that HELM does.")


if __name__ == "__main__":
    main()
