#!/usr/bin/env python3
"""Architectural comparison: HELM-MiCE vs the Euclidean DeepSeek-V3 baseline.

The HELM paper's efficiency argument for HMLA is that, like Euclidean MLA, it
"only needs to save the latent keys and values during generation ... significantly
reduces the memory footprint of the KV-cache". This script computes that
footprint from the actual model code rather than from the description, for HELM's
released shapes, and puts three attention schemes side by side:

* **dense MHA** -- what a LLaMA-style baseline caches;
* **Euclidean MLA** (DeepSeek-V3) -- ``kv_lora_rank + qk_rope_head_dim`` per token;
* **HELM HMLA** -- what this repo actually caches.

Run: ``python benchmarks/compare_architectures.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.eval.presets import PRESETS, preset_args  # noqa: E402


def cache_elements_per_token(args):
    """Per-layer KV-cache elements per token, for each attention scheme."""
    qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

    # Dense multi-head attention caches a full K and V for every head.
    dense = args.n_heads * (qk_head_dim + args.v_head_dim)

    # DeepSeek-V3 MLA caches one shared latent plus one decoupled rotary key.
    euclidean_mla = args.kv_lora_rank + args.qk_rope_head_dim

    # HELM's HMLA caches the same two objects. Every Lorentz vector spends one
    # coordinate on the time component, so each *space-like* part is one element
    # narrower than its Euclidean namesake -- the cache is marginally smaller,
    # not structurally different.
    helm_hmla = args.kv_lora_rank + (args.qk_rope_head_dim - 1)

    # What this repo cached before the latent cache existed: reconstructed
    # per-head keys and values, i.e. no better than dense.
    helm_naive = args.n_heads * (qk_head_dim + args.v_head_dim)

    return dict(dense=dense, euclidean_mla=euclidean_mla, helm_hmla=helm_hmla,
                helm_naive=helm_naive)


def main():
    seq_len, batch, bytes_per_elem = 2048, 1, 2       # bf16
    print(f"KV cache, {batch} sequence x {seq_len} tokens, bf16, all layers\n")
    header = f"{'model':16s} {'layers':>6s} {'dense MHA':>12s} {'Euclid MLA':>12s} {'HELM HMLA':>12s} {'vs dense':>9s}"
    print(header)
    print("-" * len(header))

    for name in PRESETS:
        args = preset_args(name)
        counts = cache_elements_per_token(args)
        scale = args.n_layers * seq_len * batch * bytes_per_elem / 2**20

        dense = counts["dense"] * scale
        euclid = counts["euclidean_mla"] * scale
        helm = counts["helm_hmla"] * scale
        print(f"{name:16s} {args.n_layers:6d} {dense:9.1f} MiB {euclid:9.1f} MiB "
              f"{helm:9.1f} MiB {dense / helm:8.1f}x")

    print("\nPer token, per layer, in elements:\n")
    print(f"{'model':16s} {'dense':>8s} {'Euclid MLA':>12s} {'HELM HMLA':>12s} "
          f"{'HMLA vs Euclid':>16s}")
    print("-" * 68)
    for name in PRESETS:
        counts = cache_elements_per_token(preset_args(name))
        ratio = counts["helm_hmla"] / counts["euclidean_mla"]
        print(f"{name:16s} {counts['dense']:8d} {counts['euclidean_mla']:12d} "
              f"{counts['helm_hmla']:12d} {ratio:15.3f}x")

    print("""
Reading of these numbers:

* HMLA's KV-cache saving is MLA's saving, inherited. Against dense attention it
  is large; against the Euclidean MLA that DeepSeek-V3 uses it is a rounding
  error -- one element per token per layer, because the rotary key spends a
  coordinate on the Lorentz time component. Hyperbolic geometry is not what
  shrinks the cache here; the low-rank latent is, and that is DeepSeek's idea.
* The saving is only real if the latent is what gets cached. The released HELM
  code ships its cache commented out entirely, and the first version of this
  port cached reconstructed per-head keys and values -- identical in size to
  dense attention, i.e. the entire benefit gone. `mode="latent"` is the default
  here now.""")


if __name__ == "__main__":
    main()
