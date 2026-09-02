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


#: Below this linked fraction the graph is too sparse for delta to mean anything.
#:
#: An unlinked pair sits at the constant maximum distance, so a graph that is
#: mostly unlinked is a near-uniform metric -- and every four-point sum in a
#: uniform metric is equal, which drives delta to ~0 *regardless of structure*.
#: The first run of this script reported delta 0.0009 at K=4 with 1.1% linked
#: and concluded that patches were more tree-like than tokens. They were not;
#: the graph was empty. Reporting a delta without this gate is how that happens.
MIN_LINKED = 0.15


def measure(units: Sequence, label: str, top: int, window: int, seed: int) -> Dict:
    distances, vocabulary = cooccurrence_distances(units, top, window)
    linked = (distances < 0.999).float().mean().item()
    valid = linked >= MIN_LINKED
    delta = (delta_hyperbolicity(distances, samples=200000, seed=seed)
             if valid else float("nan"))
    return {"label": label, "units": len(units), "vocabulary": len(vocabulary),
            "delta": delta, "linked_fraction": linked, "valid": valid}


def patch_profiles(sequence: Sequence[int], k: int, top: int, window: int,
                   scramble: bool = False, seed: int = 0) -> torch.Tensor:
    """Patches as aggregates of their tokens' co-occurrence profiles.

    Raw K-gram co-occurrence is unusable past K=2: the top 4-grams barely
    co-occur, the graph empties, and delta becomes vacuous. This is the
    construction the model actually uses instead -- CALM's patch embedding
    aggregates the K token representations rather than treating the K-gram as an
    atom -- so it stays dense while still asking whether the aggregate keeps the
    structure of its parts.

    Each patch becomes the mean PPMI profile of its tokens; distance is cosine
    on those profiles.
    """
    token_distances, vocabulary = cooccurrence_distances(sequence, top, window)
    index = {unit: i for i, unit in enumerate(vocabulary)}
    profiles = 1.0 - token_distances          # similarity, dense by construction

    # The control that decides whether any K-trend is linguistic. Averaging K
    # vectors pulls points toward the centroid, which can raise normalised delta
    # by arithmetic alone. `scramble` builds patches from tokens drawn at random
    # from the whole sequence instead of consecutive ones: identical averaging,
    # no adjacency. A trend that survives in the scrambled series is an artefact
    # of the mean, not a property of language.
    generator = torch.Generator().manual_seed(seed)
    rows = []
    if scramble:
        pool = [index[t] for t in sequence if t in index]
        if len(pool) < k:
            return torch.zeros((0, 0), dtype=torch.float64)
        picks = torch.randint(0, len(pool), (top, k), generator=generator)
        for row in picks:
            rows.append(profiles[[pool[i] for i in row.tolist()]].mean(0))
    else:
        for start in range(0, len(sequence) - k + 1, k):
            members = [index[t] for t in sequence[start:start + k] if t in index]
            if len(members) == k:              # only fully in-vocabulary patches
                rows.append(profiles[members].mean(0))
            if len(rows) >= top:
                break
    if len(rows) < 8:
        return torch.zeros((0, 0), dtype=torch.float64)
    matrix = torch.stack(rows)
    normed = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (1.0 - normed @ normed.T).clamp_min(0)


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

    rows = [measure(sequence, "tokens (K=1)", options.units, options.window,
                    options.seed)]
    for k in (2, options.patch, 8):
        rows.append(measure(ngrams(sequence, k), f"K-gram atoms (K={k})",
                            options.units, options.window, options.seed))

    # The construction the model uses: patches as aggregates of their tokens,
    # each paired with a scrambled control at the same K.
    for k in (1, 2, options.patch, 8):
        for scramble in (False, True):
            if k == 1 and scramble:
                continue                        # identical to K=1 by definition
            distances = patch_profiles(sequence, k, options.units,
                                       options.window, scramble=scramble,
                                       seed=options.seed)
            if distances.numel() == 0:
                continue
            suffix = ", scrambled" if scramble else ""
            rows.append({"label": f"aggregated profiles (K={k}{suffix})",
                         "units": len(sequence), "vocabulary": distances.shape[0],
                         "linked_fraction": (distances < 0.999).float().mean().item(),
                         "valid": True,
                         "delta": delta_hyperbolicity(distances, samples=200000,
                                                      seed=options.seed)})

    generator = torch.Generator().manual_seed(options.seed)
    shuffled = torch.tensor(sequence)[torch.randperm(len(sequence),
                                                     generator=generator)].tolist()
    rows.append(measure(shuffled, "tokens, order shuffled", options.units,
                        options.window, options.seed))

    print(f"{'construction':>28s} {'delta':>9s} {'linked':>8s} {'vocab':>7s}")
    for row in rows:
        delta = ("  vacuous" if not row["valid"] else f"{row['delta']:9.4f}")
        print(f"{row['label']:>28s} {delta:>9s} "
              f"{row['linked_fraction']:8.1%} {row['vocabulary']:7d}")
    print(f"\n'vacuous' = under {MIN_LINKED:.0%} of pairs linked. An unlinked "
          f"pair sits at the constant\nmaximum distance, so a mostly-unlinked "
          f"graph is a near-uniform metric, where every\nfour-point sum is "
          f"equal and delta collapses to ~0 regardless of structure.")

    token_delta = next((r["delta"] for r in rows
                        if r["label"] == "aggregated profiles (K=1)"), rows[0]["delta"])
    patch_delta = next((r["delta"] for r in rows
                        if r["label"] == f"aggregated profiles (K={options.patch})"),
                       float("nan"))
    if math.isnan(token_delta) or math.isnan(patch_delta):
        print("\nNo valid comparison: the constructions needed for it were "
              "vacuous. No verdict.")
        return
    shuffled_delta = rows[-1]["delta"]
    print(f"\nlower delta = more tree-like. Read against the shuffled control "
          f"({shuffled_delta:.4f}):\nany construction yields some delta, so only "
          f"the gap to shuffled text is structure.")
    print(f"\ntoken delta {token_delta:.4f}  ->  K={options.patch} patch delta "
          f"{patch_delta:.4f}   ratio {patch_delta / token_delta:.3f}")
    scrambled = next((r["delta"] for r in rows
                      if r["label"] == f"aggregated profiles (K={options.patch}, "
                                       f"scrambled)"), float("nan"))
    if not math.isnan(scrambled):
        print(f"scrambled control at K={options.patch}: {scrambled:.4f}  "
              f"(same averaging, no adjacency)")
        if abs(patch_delta - scrambled) < 0.15 * max(patch_delta, 1e-9):
            print("  The trend survives scrambling, so it is arithmetic -- "
                  "averaging K vectors\n  concentrates them regardless of what "
                  "they are. NOT a fact about language.")
            return
    if patch_delta > token_delta * 1.05:
        print("  Patches are LESS tree-like than their tokens, and the "
              "scrambled control does not\n  reproduce it. Patching flattens "
              "the structure HELM exploits -- the HIERARCHY.md\n  worry, in the "
              "data rather than through a model.")
    elif patch_delta < token_delta * 0.95:
        print("  Patches are MORE tree-like. The strongest case for HELM-CALM, "
              "and the one to\n  check hardest before believing.")
    else:
        print("  The structure survives aggregation: a hyperbolic HELM-CALM has "
              "the same premise\n  available that HELM does.")


if __name__ == "__main__":
    main()
