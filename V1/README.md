# V1 — Continuous Autoregressive Language Modelling on Kimi K3

An implementation of **CALM** (Continuous Autoregressive Language Models) built
on the vendored Kimi K3 stack in `src/`, written from scratch in PyTorch.

The idea in one line: **stop predicting one token at a time.** Compress every
`K` tokens into a single continuous vector with a frozen autoencoder, run the
K3 backbone autoregressively over those vectors, and predict the next vector
with a likelihood-free generative head. A sequence of `T` tokens becomes `T/K`
backbone positions, and generating `T` tokens takes `T/K` sequential steps.

```
tokens ──embed──▶ group K ──▶ project ──▶ K3 backbone ──▶ energy head
                                            (T/K positions)      │
tokens ◀──decode── frozen codec ◀───────── next latent ◀─────────┘
```

Nothing in `src/` is modified. `V1` imports `KimiBlock` — the same KDA + Gated
MLA + Stable LatentMoE + AttnRes stack the baseline uses — so any difference
between the two models is a difference in what feeds the stack and what reads
it, never in the stack itself. `backup_kimi_k3/` holds an untouched copy for
comparison.

## What is where

| file | what it is |
|---|---|
| `config.py` | `PatchAutoencoderConfig`, `EnergyHeadConfig`, `CalmK3Config`, plus tiny/small CPU factories |
| `autoencoder.py` | the token ↔ latent codec: VAE encoder, tied-weight decoder, the objective that trains them |
| `energy_head.py` | the generative head and the **energy score** that replaces cross-entropy |
| `model.py` | `CalmKimiK3` — patch embedding, the K3 backbone, the head, and the loss that ties them |
| `generation.py` | `K`-tokens-at-a-time sampling, plus the token-at-a-time baseline loop for timing |
| `metrics.py` | **BrierLM**, the one metric both a discrete and a continuous model can be scored on |
| `data.py` | a deterministic synthetic corpus (sparse Markov chain + repeated motifs) |
| `train.py` | the two stages: fit the codec, freeze it, fit the language model |
| `benchmark.py` | CALM-K3 against baseline Kimi K3 on the axes where they are comparable |
| `tests/test_v1.py` | 18 tests: shapes, freezing, propriety of the score, optimisability, generation, metrics |

## The three pieces

**The codec** (`autoencoder.py`) is a VAE over fixed patches of `K` tokens. It
emits `(mean, log_std)` rather than a point, and that is load-bearing: the head
downstream is trained against the *posterior* with a proper scoring rule, and a
scoring rule needs a distribution to score against. Encoder and decoder each run
in two stages with a squeeze/expand in the middle, so per-token layers see
tokens and per-patch layers see patches, and the decoder's output projection is
tied to the encoder's embedding.

The codec is trained first and then frozen — it defines what a latent *means*,
and a moving target would leave the energy score chasing its own tail. Its
reconstruction accuracy is a hard ceiling on the whole model: no token the codec
cannot store is reachable, whatever the backbone predicts.

**The head** (`energy_head.py`) is a sampler, not a distribution. Noise plus the
backbone's hidden state goes in, one latent comes out. There is no density, so
there is no cross-entropy and no perplexity. What replaces the likelihood is the
energy score, a strictly proper rule computable from samples alone:

```
ES(P, y) = E‖X − y‖^β − ½·E‖X − X'‖^β,     X, X' ~ P i.i.d.
```

The first term pulls samples toward the target; the second pushes them apart and
is the only thing stopping the head collapsing onto one confident guess. Drop it
and this is a regression to the posterior mean. `beta` must be in (0, 2) for
strict propriety, and below 1.0 the self-distance derivative is unbounded where
two samples coincide, so training NaNs — hence the 1.0 default and the guard in
the config.

The head is **zero-initialised**, so every sample starts identical, the
self-distance term starts at exactly zero, and the first gradients come entirely
from the cross term: the head learns where the target is before it learns how
wide it is.

**The model** (`model.py`) groups token embeddings `K` at a time, projects them
to one backbone input vector, runs `KimiBlock`, and asks the head for the next
latent. Position `p` sees patches `0..p` and predicts patch `p+1` — the same
off-by-one as next-token prediction, one patch coarser.

## Measuring it

Training losses are **not comparable**: one is a cross-entropy in nats, the
other an energy distance in latent space, and neither bounds the other. CALM-K3
has no perplexity at all.

`metrics.py` implements **BrierLM**, the CALM paper's answer — a Brier-score
estimate built from samples only, so a softmax model and a continuous one land
on the same axis:

```
brier_k = E[ 1{a₁..ₖ = y₁..ₖ} + 1{b₁..ₖ = y₁..ₖ} − 1{a₁..ₖ = b₁..ₖ} ]
BrierLM = (brier₁ · brier₂ · brier₃ · brier₄)^(1/4)
```

for two independent draws `a`, `b`. The collision term is what makes it proper:
a model that always emits the same token scores *negative*. `self_test()`
anchors the estimator against two analytically known cases — a perfect model
scores 1, a uniform model over `V` scores `V^-2.5`.

At a small scale the geometric mean is the wrong summary: it collapses to zero
the moment any single order does, and an undertrained model matches no 4-grams
at all. The benchmark therefore reports the per-order vector as well, and
`brier_1` is the number to read.

## Running it

```bash
python -m V1.train --stage both          # two-stage training on the synthetic corpus
python -m V1.benchmark --out V1/results/benchmark.json   # against baseline Kimi K3
python V1/tests/test_v1.py               # 18 tests, no pytest required
python -m V1.metrics                     # BrierLM estimator self-test
```

The corpus is generated locally and deterministically (`data.py`) because the
repository's text loaders reach for HuggingFace, which this environment cannot
reach — and because a comparison run on uniform random tokens measures nothing:
both models would be correct to output the uniform distribution and would tie by
construction. The synthetic stream has structure at two scales, local
transitions and repeated motifs, so there is something to learn.

## Results

See `results/benchmark.json` and `results/RESULTS.md` for a run and its reading.

## Scope

Everything here is CPU-scale: `d_model` 128, four routed experts, a 128-token
vocabulary, a synthetic corpus, a few hundred optimiser steps. That is enough to
show the mechanism works and to measure the structural differences —
positions per token, sequential generation steps, parameter split — and it is
nowhere near enough to say whether CALM beats a softmax head at any scale that
matters. The compression argument is architectural and transfers; the quality
numbers do not.
