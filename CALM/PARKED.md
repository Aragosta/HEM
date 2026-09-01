# Parked: end-to-end efficiency measurement at the released HELM shapes

Specified but **not run**. Parked because it measures a projection we already
have component-wise, and because the scale analysis in `NEXT.md` §3 changes which
shape it should be run at.

## What it would measure

Every efficiency number for HELM-CALM so far is either a component measurement
(the head in isolation — 12x fewer parameters, 125x smaller activation, 3.03x
faster) or arithmetic (sequence shortens by K, so attention scores fall by K²).
Neither is a whole-model number.

This would build an untrained HELM-CALM at a released shape and time a full
training step against discrete HELM at the same shape:

| | discrete HELM | HELM-CALM K=4 |
| --- | --- | --- |
| forward + backward, ms | | |
| peak RSS | | |
| dispatched aten ops | | |

and separately, generation: tokens/second at a fixed output length, where CALM
gets both the K-fold reduction in autoregressive steps and the latent KV cache.

## Why it is worth running eventually

The head and sequence savings do not compose trivially. The energy head runs
`num_samples=8` forward passes per position, and its activation is
`8 x tokens x latent` where the vocab head's was `tokens x 128256` — a large
reduction, but the factor-of-8 sample multiplier applies to the head's *compute*
as well, and only a measurement settles where that lands. The frozen autoencoder
encoder also runs every step (no gradient, but real time).

## How to run it

```bash
python CALM/experiments/bench_helm_calm.py \
    --preset helm_mice_1B --patch 4 --batch-size 2 --seq-len 2048
```

That script does not exist yet. It should reuse:

* `benchmarks/bench_helm_mice.py::timeit`, `peak_memory`, `count_ops` — the
  harness is already written and handles the complex-buffer casting trap;
* `CALM/experiments/stage2a_patching.py::LorentzPatchEmbedding` and
  `patched_hidden` — the model path;
* `CALM/estimate_helm_calm.py::CalmHead` — the head.

Peak memory should be measured as **process RSS in a subprocess**, not
`torch.profiler`'s per-op figure. `docs/UPGRADES.md` records why: the profiler
reports the largest single allocation, which for the vocab head happens to look
right and for a chunked path does not.

## Run it at the 1B preset, not 120M

`NEXT.md` §3 works out that at the 120M shape the frozen autoencoder is 53% of
the deployed system and HELM-CALM is *larger* in total than plain HELM. A
whole-model efficiency measurement at 120M would therefore describe an operating
point nobody should use. The 1B preset is where the economics resemble CALM's
own, and is the shape to measure.
