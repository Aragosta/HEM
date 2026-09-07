#!/usr/bin/env python3
"""T11, the part that fits: the compressibility/mixing tradeoff, measured on the
attention graphs themselves.

The full T11 needs mask generators, synthetic multi-hop tasks and training runs.
But the *tension* the experiment is built around is a property of the graphs, not
of the models, and it can be measured exactly and in seconds:

* a **compressible** graph is modular and clustered -- and clustered graphs have
  long paths, so information mixes slowly;
* an **expander** mixes in a few hops -- and expanders are built from random
  edges, which is precisely what cannot be compressed.

So compressibility and mixing pull against each other, and no amount of training
changes that. What training decides is where on the frontier a task wants to sit.
This module draws the frontier. It does NOT test which point wins on a task, and
it must not be read as doing so.

**Every mask carries the same number of edges.** Density is the confound that
would otherwise explain every difference, so it is removed by construction rather
than controlled for afterwards, and the runner asserts it.

Metrics, per mask:

* **compressed adjacency bytes** -- zlib over the packed causal adjacency
  bitstring. A proxy for description length, not the rate-distortion optimum;
  what matters is that it is the same proxy for every mask.
* **reachability within `depth` hops** -- the fraction of (query, earlier key)
  pairs joined by a path of at most `depth` steps. This is the one that matters
  for a transformer: an L-layer model can only move information L hops, so a
  pair unreachable in L hops is a dependency the model structurally cannot
  represent, whatever it learns.
* **spectral gap** of the symmetrised normalised Laplacian -- the standard
  mixing rate.
* **mean path length**, **clustering**, **degree CV** -- the compressibility
  signature from the network literature (high transitivity, heterogeneous
  degrees).

**Predictions, registered before the run:**

Q1. Compressed size rises monotonically with the rewiring probability `p`.
    Randomness is incompressible; this is close to definitional and is the
    sanity check on the measure.
Q2. Reachability within 4 hops also rises with `p`, and rises FASTER at small
    `p` than compressed size does. That gap is the small-world claim, and it is
    what would make an interior `p` the right choice for a 4-layer model.
Q3. `bigbird` sits near the small-world curve rather than off it -- its random
    edges are the shortcuts, so it is the same object under another name.
Q4. `window+global` achieves high reachability at low compressed size, because a
    global token is a hub: cheap to describe, and it shortens every path. If
    this holds, the standard design is already close to the frontier and T11's
    interesting question is narrower than it looked.

Usage::

    python CALM/suite/graphs.py --seq-len 128 --window 16 --depth 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import zlib
from collections import deque
from pathlib import Path
from typing import Dict, List, Set

Graph = List[Set[int]]          # graph[q] = keys q attends to, all <= q


def window_mask(n: int, w: int) -> Graph:
    return [{k for k in range(max(0, q - w), q + 1)} for q in range(n)]


def edge_count(graph: Graph) -> int:
    return sum(len(row) for row in graph)


def _trim_or_grow(graph: Graph, target: int, rng: random.Random) -> Graph:
    """Force an exact edge budget, so density never explains a difference."""
    while edge_count(graph) > target:
        q = rng.randrange(1, len(graph))
        removable = [k for k in graph[q] if k != q]      # never drop self
        if removable:
            graph[q].discard(rng.choice(removable))
    while edge_count(graph) < target:
        q = rng.randrange(1, len(graph))
        options = [k for k in range(q) if k not in graph[q]]
        if options:
            graph[q].add(rng.choice(options))
    return graph


def small_world(n: int, w: int, p: float, target: int, seed: int) -> Graph:
    """Watts-Strogatz on a causal graph: rewire a fraction p of window edges."""
    rng = random.Random(seed)
    graph = window_mask(n, w)
    for q in range(n):
        for k in list(graph[q]):
            if k != q and rng.random() < p:
                options = [j for j in range(q) if j not in graph[q]]
                if options:
                    graph[q].discard(k)
                    graph[q].add(rng.choice(options))
    return _trim_or_grow(graph, target, rng)


def window_global(n: int, w: int, g: int, target: int, seed: int) -> Graph:
    rng = random.Random(seed)
    graph = window_mask(n, w)
    for q in range(n):
        graph[q].update(j for j in range(min(g, q + 1)))
    return _trim_or_grow(graph, target, rng)


def bigbird(n: int, w: int, g: int, r: int, target: int, seed: int) -> Graph:
    rng = random.Random(seed)
    graph = window_global(n, w, g, target, seed)
    for q in range(n):
        for _ in range(r):
            if q:
                graph[q].add(rng.randrange(q))
    return _trim_or_grow(graph, target, rng)


def strided_hubs(n: int, w: int, stride: int, target: int, seed: int) -> Graph:
    """Hubs spread through the sequence, not parked at the start.

    The distinction matters only in a causal graph, and it is decisive there. A
    hub at position h can only ever carry information about positions <= h, so a
    path routed through it is a dead end for any target above h. Hubs at 0..3 --
    the Longformer/StreamingLLM "global tokens" or attention sinks -- therefore
    shorten paths to tokens that were already reachable and relay nothing.
    Spreading the same number of hubs across the sequence restores the relay.
    """
    rng = random.Random(seed)
    graph = window_mask(n, w)
    for q in range(n):
        graph[q].update(h for h in range(0, n, stride) if h <= q)
    return _trim_or_grow(graph, target, rng)


def dilated(n: int, w: int, target: int, seed: int) -> Graph:
    """Powers-of-two offsets: log(n) shortcuts per token, deterministic."""
    rng = random.Random(seed)
    graph = window_mask(n, w)
    for q in range(n):
        offset = 1
        while offset <= q:
            graph[q].add(q - offset)
            offset *= 2
    return _trim_or_grow(graph, target, rng)


def modular(n: int, block: int, target: int, seed: int) -> Graph:
    """Hierarchical blocks: dense inside a block, one link to the block before."""
    rng = random.Random(seed)
    graph: Graph = [set() for _ in range(n)]
    for q in range(n):
        start = (q // block) * block
        graph[q].update(k for k in range(start, q + 1))
        if start:
            graph[q].add(start - 1)
    return _trim_or_grow(graph, target, rng)


def compressed_bytes(graph: Graph, n: int) -> int:
    bits = bytearray()
    for q in range(n):
        for k in range(n):
            bits.append(1 if k in graph[q] else 0)
    packed = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        packed.append(byte)
    return len(zlib.compress(bytes(packed), 9))


def reachability(graph: Graph, n: int, depth: int) -> float:
    """Fraction of (query, earlier key) pairs joined by a path of <= depth hops.

    Directed backwards, which is how information actually travels: a query reads
    its keys, those keys read theirs, and after `depth` layers the query has seen
    everything within `depth` hops. Pairs outside that radius are dependencies
    the architecture cannot represent at this depth.
    """
    total = reached = 0
    for source in range(n):
        seen = {source}
        frontier = deque([(source, 0)])
        while frontier:
            node, distance = frontier.popleft()
            if distance == depth:
                continue
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append((neighbour, distance + 1))
        total += source + 1
        reached += len(seen & set(range(source + 1)))
    return reached / total


def reach_by_distance(graph: Graph, n: int, depth: int,
                      bands=((1, 4), (5, 16), (17, 48), (49, 127))) -> Dict[str, float]:
    """Reachability split by separation |q - k|, which is where the mechanism is.

    A causal graph has an asymmetry an undirected one does not: to reach the
    token immediately before you, a DIRECT edge is the only option, because any
    detour would have to pass through a position strictly between them and there
    is none. Near pairs therefore have exactly one route and far pairs have many.
    That makes local band edges irreplaceable and long-range edges substitutable,
    which is why deleting the band (p=1) destroys short-range reachability while
    leaving long-range reachability almost untouched.
    """
    hits = {band: [0, 0] for band in bands}
    for source in range(n):
        seen = {source}
        frontier = deque([(source, 0)])
        while frontier:
            node, distance = frontier.popleft()
            if distance == depth:
                continue
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append((neighbour, distance + 1))
        for key in range(source):
            separation = source - key
            for band in bands:
                if band[0] <= separation <= band[1]:
                    hits[band][1] += 1
                    hits[band][0] += key in seen
    return {f"reach_{a}_{b}": h[0] / max(h[1], 1) for (a, b), h in hits.items()}


def spectral_gap(graph: Graph, n: int) -> float:
    import torch
    adjacency = torch.zeros(n, n)
    for q in range(n):
        for k in graph[q]:
            adjacency[q, k] = 1.0
            adjacency[k, q] = 1.0
    degree = adjacency.sum(1).clamp_min(1e-9)
    normalised = adjacency / degree.sqrt().unsqueeze(1) / degree.sqrt().unsqueeze(0)
    laplacian = torch.eye(n) - normalised
    values = torch.linalg.eigvalsh(laplacian)
    return values[1].item()


def clustering(graph: Graph, n: int) -> float:
    undirected = [set() for _ in range(n)]
    for q in range(n):
        for k in graph[q]:
            if k != q:
                undirected[q].add(k)
                undirected[k].add(q)
    scores = []
    for node in range(n):
        neighbours = undirected[node]
        if len(neighbours) < 2:
            continue
        links = sum(1 for a in neighbours for b in neighbours
                    if a < b and b in undirected[a])
        scores.append(2 * links / (len(neighbours) * (len(neighbours) - 1)))
    return statistics.mean(scores) if scores else 0.0


def degree_cv(graph: Graph, n: int) -> float:
    degrees = [len(row) for row in graph]
    mean = statistics.mean(degrees)
    return statistics.stdev(degrees) / mean if mean else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="CALM/suite/graphs_results.json")
    options = parser.parse_args()
    n, w, depth = options.seq_len, options.window, options.depth

    budget = edge_count(window_mask(n, w))
    print("T11 (graph half)  compressibility against mixing, at a FIXED edge budget")
    print(f"    seq_len {n}, window {w}, depth {depth}, "
          f"{budget:,} edges in every mask "
          f"({100 * budget / (n * (n + 1) / 2):.1f}% of the causal upper triangle)\n")

    masks = {f"p={p}": small_world(n, w, p, budget, options.seed)
             for p in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)}
    masks["window+global"] = window_global(n, w - 4, 4, budget, options.seed)
    masks["bigbird"] = bigbird(n, w - 6, 3, 3, budget, options.seed)
    masks["modular"] = modular(n, 2 * w, budget, options.seed)
    masks["hubs /16"] = strided_hubs(n, w - 5, 16, budget, options.seed)
    masks["hubs /32"] = strided_hubs(n, w - 4, 32, budget, options.seed)
    masks["dilated"] = dilated(n, 4, budget, options.seed)

    print(f"    {'mask':>14s} {'zlib B':>7s} {'reach':>7s} "
          f"{'near':>7s} {'window':>7s} {'mid':>7s} {'far':>7s} {'gap':>7s}")
    rows = []
    for name, graph in masks.items():
        assert edge_count(graph) == budget, f"{name} broke the edge budget"
        row = {
            "mask": name,
            "edges": edge_count(graph),
            "zlib_bytes": compressed_bytes(graph, n),
            "reach": reachability(graph, n, depth),
            **reach_by_distance(graph, n, depth),
            "spectral_gap": spectral_gap(graph, n),
            "clustering": clustering(graph, n),
            "degree_cv": degree_cv(graph, n),
        }
        rows.append(row)
        print(f"    {name:>14s} {row['zlib_bytes']:7d} {row['reach']:7.3f} "
              f"{row['reach_1_4']:7.3f} {row['reach_5_16']:7.3f} "
              f"{row['reach_17_48']:7.3f} {row['reach_49_127']:7.3f} "
              f"{row['spectral_gap']:7.4f}")

    sweep = [r for r in rows if r["mask"].startswith("p=")]
    sizes = [r["zlib_bytes"] for r in sweep]
    reaches = [r["reach"] for r in sweep]
    print()
    q1 = all(b >= a - 1 for a, b in zip(sizes, sizes[1:]))
    print(f"    Q1 compressed size rises with p: {'HOLDS' if q1 else 'FAILS'} "
          f"({sizes[0]} -> {sizes[-1]} bytes)")
    # Efficiency: reachability bought per compressed byte, relative to p=0.
    gains = [(r["reach"] - sweep[0]["reach"]) /
             max(r["zlib_bytes"] - sweep[0]["zlib_bytes"], 1) for r in sweep[1:]]
    best = sweep[1:][gains.index(max(gains))]["mask"] if gains else "n/a"
    print(f"    Q2 reachability per compressed byte is best at {best} "
          f"(reach {sweep[0]['reach']:.3f} at p=0 -> {sweep[-1]['reach']:.3f} at p=1)")
    hub = next(r for r in rows if r["mask"] == "hubs /16")
    wg = next(r for r in rows if r["mask"] == "window+global")
    print(f"    Q4 window+global reach {wg['reach']:.4f} vs the SAME hub count "
          f"spread through the sequence {hub['reach']:.4f} -- hub PLACEMENT, "
          f"not the hub idea")
    bb = next(r for r in rows if r["mask"] == "bigbird")
    print(f"    Q3 bigbird reach {bb['reach']:.4f} at {bb['zlib_bytes']} B; "
          f"nearest sweep point "
          f"{min(sweep, key=lambda r: abs(r['zlib_bytes'] - bb['zlib_bytes']))['mask']}")
    print(f"    Q4 window+global reach {wg['reach']:.4f} at {wg['zlib_bytes']} B "
          f"vs p=1 reach {sweep[-1]['reach']:.4f} at {sweep[-1]['zlib_bytes']} B")
    print("\n    Graph structure only. Which point a task wants is not tested here.")

    if options.out:
        Path(options.out).write_text(json.dumps(
            {"config": vars(options), "budget": budget, "rows": rows}, indent=1))
        print(f"    wrote {options.out}")


if __name__ == "__main__":
    main()
