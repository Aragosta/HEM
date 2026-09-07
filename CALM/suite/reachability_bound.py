#!/usr/bin/env python3
"""TOPOLOGY.md 3.1: does the reachability bound predict what a model can learn?

Every topology result in this project so far is a statement about graphs. This
is the experiment that connects one to a trained model, and it is worth more
than an accuracy comparison because **the theory makes a point prediction with
no free parameters.**

Following arXiv:2606.02680 (Theorem 2): let `R_L(t)` be the structural
dependency closure of position `t` after `L` layers -- the positions reachable
by following attention edges backwards. If two inputs agree on `R_L(t)` then
`h_t` is *identical*, because positionwise MLPs, normalisation and residuals
cannot add source positions. So build a task whose answer lives at exactly one
position `s`, and:

* **`s` unreachable** -> `h_t` cannot depend on the answer. Top-1 accuracy is at
  most `1/K` and cross-entropy at least `log K`, for ANY parameters. Not a
  tendency, an upper bound.
* **`s` reachable** -> the mask permits a solution, and whether the model finds
  it is a learning question, not a structural one.

The task: a sequence of random filler tokens with one class token `c` (of `K`)
placed at position `s`, and `c` again at position `t + 1`. Standard next-token
prediction at `t` is then exactly a copy from `s`, and the loss is taken at that
position only, so nothing else competes for capacity. Distances are chosen to
straddle each mask's reachability boundary.

Six masks at 4 layers and seq 128, each carrying 2040 edges per layer except
where noted:

* `full`      -- control; everything reachable, and the learning ceiling.
* `window16`  -- pure local band; far pairs unreachable (measured reach 0.362).
* `sw0.03`    -- small-world; reach 1.000.
* `winglobal` -- window + sinks at 0-3; far reach 0.098, the worst measured.
* `hubs16`    -- landmarks every 16; reach 1.000.
* `layered`   -- 3x window w=3 + 1 landmark layer, 2718 edges TOTAL (0.33x).

**Predictions, registered before the run:**

R1. Every (mask, distance) cell whose source is unreachable scores at or below
    `1/K`, within binomial noise. **This is the falsifiable core**: a single
    unreachable cell scoring above the bound means the reachability computation
    in `graphs.py` is wrong, and every topology claim in `TOPOLOGY.md` falls
    with it.
R2. `full` solves every distance. If it does not, the task is unlearnable at
    this budget and no other row can be read.
R3. Reachable cells beat their bound but do NOT all reach `full`'s accuracy, and
    the shortfall tracks influence `Phi` rather than reachability -- which is
    the separation between "a path exists" and "the path carries anything", and
    is what 3.2 re-analyses from these same runs.
R4. `layered`, at a third of the edge budget, is not worse than `sw0.03`. It has
    both full reach and the higher far-band influence, so if it loses, influence
    at initialisation is not the right predictor of what training can use.

Usage::

    python CALM/suite/reachability_bound.py --seeds 0,1,2
    python CALM/suite/reachability_bound.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "suite"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from graphs import (edge_count, full_mask, hub_layer,  # noqa: E402
                    small_world, window_global, window_mask)
from hybrid import HybridDecoder  # noqa: E402

FILLER = 64          # filler vocabulary
DISTANCES = (2, 8, 24, 60, 100)


def mask_families(n: int, layers: int, seed: int) -> Dict[str, List]:
    budget = edge_count(window_mask(n, 16))
    return {
        "full": [full_mask(n)] * layers,
        "window16": [window_mask(n, 16)] * layers,
        "sw0.03": [small_world(n, 16, 0.03, budget, seed + i) for i in range(layers)],
        "winglobal": [window_global(n, 12, 4, budget, seed)] * layers,
        "hubs16": [hub_layer(n, 11, 16)] * layers,
        "layered": [window_mask(n, 3)] * (layers - 1) + [hub_layer(n, 0, 8)],
    }


def to_tensor(graphs: List, n: int) -> List[torch.Tensor]:
    out = []
    for graph in graphs:
        mask = torch.zeros(n, n, dtype=torch.bool)
        for q in range(n):
            for k in graph[q]:
                mask[q, k] = True
            mask[q, q] = True
        out.append(mask)
    return out


def closure(graphs: List, n: int) -> List[set]:
    """R_L(t) for every t: positions reachable by walking edges backwards.

    Layers are traversed in reverse, because the last layer acts last and is
    therefore the first hop away from the output.
    """
    sets = []
    for q in range(n):
        seen = {q}
        for graph in reversed(graphs):
            nxt = set(seen)
            for node in seen:
                nxt |= graph[node]
            seen = nxt
        sets.append(seen)
    return sets


def make_batch(batch: int, n: int, k_classes: int, distance: int,
               generator: torch.Generator) -> tuple:
    """Filler everywhere, class token at `s`, the same token again at `t + 1`."""
    tokens = torch.randint(k_classes, k_classes + FILLER, (batch, n + 1),
                           generator=generator)
    targets = torch.randint(0, k_classes, (batch,), generator=generator)
    t = torch.randint(distance, n - 1, (batch,), generator=generator)
    s = t - distance
    rows = torch.arange(batch)
    tokens[rows, s] = targets
    tokens[rows, t + 1] = targets
    return tokens, t, s, targets


def evaluate(model, n, k_classes, distance, batches, batch, device, generator):
    """Top-1 accuracy and cross-entropy at the probe position only."""
    hits = total = 0
    nats = 0.0
    per_position = {}
    with torch.no_grad():
        for _ in range(batches):
            tokens, t, s, targets = make_batch(batch, n, k_classes, distance,
                                               generator)
            tokens = tokens.to(device)
            logits = model.logits(tokens[:, :-1])
            rows = torch.arange(tokens.size(0))
            probe = logits[rows, t][:, :k_classes].float()
            loss = torch.nn.functional.cross_entropy(
                probe, targets.to(device), reduction="sum")
            nats += loss.item()
            correct = probe.argmax(-1).cpu() == targets
            hits += int(correct.sum())
            total += targets.numel()
            for position, ok in zip(t.tolist(), correct.tolist()):
                bucket = per_position.setdefault(position, [0, 0])
                bucket[0] += int(ok)
                bucket[1] += 1
    return hits / total, nats / total, per_position


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--classes", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="CALM/suite/reachability_results.json")
    o = p.parse_args()
    if o.smoke:
        o.steps, o.seeds, o.eval_batches = 30, "0", 2

    device = torch.device(o.device)
    n, K = o.seq_len, o.classes
    vocab = K + FILLER
    seeds = [int(v) for v in o.seeds.split(",")]
    bound = 1.0 / K
    started = time.time()

    print("REACHABILITY BOUND  does R_L(t) predict what a model can learn?")
    print(f"    seq {n}, {o.layers} layers, dim {o.dim}, K={K} classes, "
          f"vocab {vocab}, {o.steps} steps, seeds {seeds}")
    print(f"    chance = 1/K = {bound:.4f}, and log K = {math.log(K):.3f} nats "
          f"is the CE floor for an unreachable source\n")

    families = mask_families(n, o.layers, 0)
    coverage = {}
    for name, graphs in families.items():
        sets = closure(graphs, n)
        coverage[name] = {
            d: statistics.mean(float((q - d) in sets[q])
                               for q in range(d, n - 1))
            for d in DISTANCES}
    print(f"    {'mask':>10s} {'edges':>7s} " +
          " ".join(f"d={d:<4d}" for d in DISTANCES) + "   <- fraction of probe "
          "positions whose source is reachable")
    for name, graphs in families.items():
        total_edges = sum(edge_count(g) for g in graphs)
        print(f"    {name:>10s} {total_edges:7d} " +
              " ".join(f"{coverage[name][d]:6.3f}" for d in DISTANCES))
    print()

    rows: List[Dict] = []
    for seed in seeds:
        for name, graphs in families.items():
            torch.manual_seed(seed)
            model = HybridDecoder(vocab, dim=o.dim, layers=o.layers,
                                  heads=max(o.dim // o.head_dim, 1),
                                  head_dim=o.head_dim, kv_latent=None,
                                  ffn="dense", max_seq_len=n).to(device)
            model.set_masks([m.to(device) for m in to_tensor(graphs, n)])
            optimizer = torch.optim.AdamW(model.parameters(), lr=o.lr,
                                          weight_decay=0.01)
            generator = torch.Generator().manual_seed(seed)
            model.train()
            for step in range(o.steps):
                distance = DISTANCES[step % len(DISTANCES)]
                tokens, t, _, targets = make_batch(o.batch, n, K, distance,
                                                   generator)
                tokens = tokens.to(device)
                logits = model.logits(tokens[:, :-1])
                probe = logits[torch.arange(tokens.size(0)), t][:, :K]
                loss = torch.nn.functional.cross_entropy(
                    probe.float(), targets.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()

            line = []
            for distance in DISTANCES:
                accuracy, ce, _ = evaluate(model, n, K, distance,
                                           o.eval_batches, o.batch, device,
                                           generator)
                rows.append({"seed": seed, "mask": name, "distance": distance,
                             "accuracy": accuracy, "ce": ce,
                             "reachable_frac": coverage[name][distance]})
                line.append(f"{accuracy:6.3f}")
            print(f"    seed {seed} {name:>10s}  " + " ".join(line))

    print("\n" + "=" * 74)
    print("R1  every cell with an UNREACHABLE source must sit at or below "
          f"1/K = {bound:.4f}")
    violations = []
    print(f"    {'mask':>10s} {'dist':>5s} {'reach':>6s} {'acc':>7s} {'ce':>7s} "
          f"{'verdict':>10s}")
    for name in families:
        for distance in DISTANCES:
            cells = [r for r in rows if r["mask"] == name
                     and r["distance"] == distance]
            reach = cells[0]["reachable_frac"]
            accuracy = statistics.mean(c["accuracy"] for c in cells)
            ce = statistics.mean(c["ce"] for c in cells)
            if reach == 0.0:
                # A cell with no reachable source. The bound is exact.
                ok = accuracy <= bound * 2.5      # binomial slack at this n
                verdict = "at bound" if ok else "VIOLATION"
                if not ok:
                    violations.append((name, distance, accuracy))
                print(f"    {name:>10s} {distance:5d} {reach:6.3f} "
                      f"{accuracy:7.3f} {ce:7.3f} {verdict:>10s}")
    print()
    if violations:
        print("    R1 FAILS. An unreachable source was predicted above chance, "
              "which means the\n    reachability computation is wrong and every "
              "topology claim resting on it\n    falls with it:")
        for name, distance, accuracy in violations:
            print(f"      {name} d={distance}: {accuracy:.3f} > {bound:.4f}")
    else:
        print("    R1 HOLDS. No unreachable cell beat chance, so the structural "
              "bound is real\n    and the graph analysis predicts a hard "
              "ceiling on what training can find.")

    full_rows = {d: statistics.mean(r["accuracy"] for r in rows
                                    if r["mask"] == "full" and r["distance"] == d)
                 for d in DISTANCES}
    print(f"\nR2  full attention by distance: " +
          " ".join(f"d={d}:{full_rows[d]:.3f}" for d in DISTANCES))
    print("    " + ("HOLDS -- the task is learnable, so the other rows can be read"
                    if min(full_rows.values()) > 0.5 else
                    "FAILS -- full attention cannot solve this at this budget, "
                    "and no row below\n    is interpretable"))

    print("\nR3/R4  reachable cells, mean accuracy by mask "
          "(only distances with reach = 1)")
    for name in families:
        cells = [r for r in rows if r["mask"] == name
                 and r["reachable_frac"] >= 0.999]
        if cells:
            total_edges = sum(edge_count(g) for g in families[name])
            print(f"    {name:>10s} {total_edges:7d} edges  "
                  f"{statistics.mean(c['accuracy'] for c in cells):6.3f} "
                  f"over {len(cells) // len(seeds)} distances")
    print("=" * 74)
    print(f"\nwall clock {(time.time() - started) / 60:.1f} min")

    if o.out:
        Path(o.out).write_text(json.dumps(
            {"config": vars(o), "coverage": {k: {str(d): v for d, v in c.items()}
                                             for k, c in coverage.items()},
             "rows": rows}, indent=1))
        print(f"wrote {o.out}")


if __name__ == "__main__":
    main()
