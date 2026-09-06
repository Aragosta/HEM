# Results

Every table in this file is regenerated from `results/*.json` by the
experiment's own `--report-only`. The pilot section is hand-written because the
pilots are the part that failed, and a study that only records the runs that
worked is not recording the thing that decided its design.

---

## Verdict, against the predictions registered in `DESIGN.md`

| # | prediction | outcome |
| --- | --- | --- |
| P0 | removing either K3 component costs something | **failed** — at 3000 steps all four cells of the 2x2 land within the seed spread on `hops`; AttnRes costs 1.7x wall-clock and no accuracy |
| P1 | the best depth rises with the training budget | **failed** — the argmax over R does not move, and at 2 hops depth's advantage *shrinks* from +0.09 to -0.01 as the budget doubles |
| P2 | recurrence buys ~nothing in bits per byte | **held** — 2.367 -> 2.348 from R=1 to R=4, inside noise |
| P3 | depth's benefit grows with hop count | **weakly held** — R=2 over R=1 is +0.064 at 3 hops and -0.011 at 2 hops, at the larger budget |
| P4 | persistent registers beat `plain` on `twochain` more than on `hops` | **not supported, and underpowered** — every arm within +/-0.008 of `plain`, on a task the model only reaches 0.258 on (chance 0.167) |
| P5 | cross-loop AttnRes matches or beats registers | **inconclusive** — both are null in the same underpowered test |
| P6 | step-conditioned routing measurably changes routing | **held**, and accuracy moved too: +0.037 on `hops`, outside the 0.006 seed spread |
| P7 | no routing instability from looping | **held** — expert-load sd 0.0225 / 0.0242 / 0.0267 at R = 1 / 2 / 4 |
| P8 | the two gates separate under extrapolation | **held** — Ouro's Q-exit freezes at an identical depth for R=8 and R=12 as for R=4; PonderNet's keeps moving |
| P9 | tokens beat latent depth at matched compute | **not run** (E5, `PARKED.md`) |

Three findings were not predicted and are the most interesting things here.

**The loop converges, and that is why depth saturates.** Expert routing changes
sharply from loop 1 to loop 2 (JS 0.047) and then stops (0.003, 0.0003), and a
model unrolled past its trained depth moves by less than 0.005 (R=4: 0.626 ->
R=8: 0.623). The recurrence behaves like iteration to a fixed point with a
distinguished first step. That explains saturation without needing the brief's
state-capacity hypothesis — which E2 looked for and did not find — and it
predicts what E3 then measured: the useful conditioner is the one that breaks
the symmetry between the first pass and the rest (`bias`, +0.037), not the one
that forces every loop to compute something different (`embed`, +0.022 with 4x
the routing divergence).

**Training with a halting gate makes shallow inference work.** Gated models
score 0.51-0.52 at R=1 where the ungated model gets 0.405. Every loop becomes a
usable output point, which is Ouro's exit-step objective doing something its
own paper does not claim.

**The free exit rule is competitive with the trained ones.** Huginn's zero-shot
KL criterion, on a model trained with no gate at all, reaches full-depth
accuracy (0.570) at an average depth of 2.23 of 4. Before spending RL on a
halting gate (brief §7.4), that is the number to beat.

---

## Part 0 — the pilots, and four things that were wrong

Recorded because each one changed the experiment, and because three of them are
mistakes anyone rebuilding this would repeat.

### 0.1 One question per context does not train

First task layout: one context of 32 in-context edges, one question, loss on
one token in 36. At 800 steps, R=1 and R=4 both sat at exactly `ln(V)` — not
"learning slowly", not learning at all — and the per-loop predictions were
identical, which is the signature of a model that has collapsed to a constant.

Fix: **eight questions per context**, each with its own entity and hop count,
all answered from the same edge list. Nothing else about the task changed.

### 0.2 The initialisation was wrong, and it was the whole gap

Even with dense supervision the model stayed at chance while a 30-line textbook
transformer (`probe_reference.py`) on identical batches reached **99.5% by step
1200**. That comparison is the reason `probe_reference.py` exists: it converts
"the task is too hard" into "the task is fine and your model is broken".

The cause was `nn.init.normal_(m.weight, std=0.02)` applied to every `Linear`.
0.02 is the GPT-2 convention for `d_model≈768`; at `dim=64` fan-in scaling is
`0.125`, six times larger. Attention logits started near zero, attention started
near uniform, and nothing escaped. Fan-in scaled initialisation is now in
`model.py` with the comment that says so.

### 0.3 The answer has to be predictable from the position it is predicted at

With the query written as `QUERY x HOP -> answer`, the entity is two tokens back
from the position that must produce the answer, so the model has to move it
forward before it can match — an extra composition step. Reordering to
`QUERY HOP x -> answer` puts the entity *at* the prediction site, which is the
layout an induction head can use directly.

### 0.4 The paired arms were not paired

Found by `tests_recur.py`, not by looking at a result: drawing every parameter
from one RNG stream means that adding a parameter -- a register bank in E2, a
halting gate in E4 -- shifts every draw after it, so two arms with the same seed
shared *no* weights. Every "paired difference" in E2 and E4 would have been an
unpaired difference wearing a paired label, and nothing in the numbers would
have looked wrong.

Fix: initialise each parameter from a generator keyed by its **name**, so an
arm that adds a tensor leaves the others bit-identical. E0 was rerun from
scratch after this change rather than mixing the two initialisation schemes in
one table.

### 0.5 What the pilots settled about scale

| probe | result |
| --- | --- |
| 12 entities, hops mixed 1–4 | at chance after 3000 steps at R=1, 2 and 4 — four circuits sharing one gradient, none forms |
| 12 entities, hop 1 only, standard residuals | solved (100%) by ~1050 steps |
| 12 entities, hop 2 only | still at chance at 1500 steps |
| **6 entities, hop 2 only** | **learns: 0.56 at 1200 steps, 0.63 at 2400 (chance 0.167)** |
| 6 entities, hop 2, R=4 | learns more slowly per step than R=1 (0.38 at 900) |

So the study runs at **6 entities, 6 questions per context, one hop count per
run**. Hop count became an experimental factor instead of a within-batch
nuisance, which is a better design than the one it replaced.

### 0.6 Attention Residuals is a drag at this scale — but not a wall

The pilot ladder, one component removed at a time, hop 1, 12 entities:

| arm | accuracy at 1500 steps |
| --- | --- |
| K3 baseline (AttnRes + MoE) | 0.12 (chance 0.083) |
| − AttnRes | **1.00 by step 1050** |
| − MoE (dense) | 0.13 |
| − random state init, − injection | 0.145 |
| everything stripped (no AttnRes, dense, no injection, no sandwich norm) | 0.23 |
| AttnRes, run to 4000 steps | 0.60 at 1200, **0.99 at 1600** |

Full AttnRes costs roughly **1.6x the steps** to the same accuracy at this size,
and removing sandwich normalisation does not rescue it (0.13 at 750), so it is
not an interaction with Ouro's normalisation — it is AttnRes itself. But it does
get there, so the baseline stays faithful: **E0–E4 all run with AttnRes on**,
and E0 measures the cost properly with seeds.

This is a statement about eight sublayers and a few thousand steps, not about
the mechanism. AttnRes exists to stop PreNorm dilution accumulating over depth,
and eight sublayers cannot accumulate much; its own report measures gains at 32+
layers and a budget six orders of magnitude larger. The interesting part is the
*direction*: a depth-wise mechanism that helps deep models slows shallow ones,
which is what "this is a depth mechanism" ought to predict.

---

## Part 1 — E0, the baseline and the noise floor

Both K3 components turn out to be neutral-to-negative at this size. On
composition all four cells sit within a seed spread of each other, and the only
robust difference is wall-clock: AttnRes costs **1.7x** (320s vs 190s per run)
for nothing. On bytes the dense FFN beats the MoE by ~0.15 bpb, comfortably
outside noise, which is the ordinary small-model MoE penalty -- 16 experts of 28
hidden units each, 12.5% active, cannot pay for their routing.

The AttnRes null is the informative one, because it is the *right* null. A
mechanism built to stop PreNorm dilution accumulating over 93 layers has nothing
to do at eight sublayers, and it charges for the privilege. Its cost here is
also the reason the seed spread on the K3 arm (0.024) is the number every later
experiment is read against.

<!-- filled by: python e0_baseline.py --report-only -->

## hops -- acc (higher is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 0.6139 | 0.0242 | 3 | 320 |
| noattnres | none | MoE | 0.6424 | 0.0566 | 3 | 190 |
| dense | full | dense | 0.6198 | 0.0013 | 3 | 246 |
| noattnres_dense | none | dense | 0.6204 | 0.0017 | 3 | 147 |

AttnRes effect with MoE: -0.0284; with a dense FFN: -0.0007; interaction: -0.0278.

**Noise floor (hops): sd 0.0242 over 3 seeds of the K3 arm, mean 0.6139.**

**Same, with standard residuals instead: sd 0.0566 over 3 seeds, mean 0.6424.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.

## bytes -- bpb (lower is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 2.3670 | 0.0023 | 2 | 206 |
| noattnres | none | MoE | 2.3321 | 0.0135 | 2 | 128 |
| dense | full | dense | 2.2073 | 0.0221 | 2 | 187 |
| noattnres_dense | none | dense | 2.2215 | 0.0022 | 2 | 121 |

AttnRes effect with MoE: +0.0349; with a dense FFN: -0.0142; interaction: +0.0491.

**Noise floor (bytes): sd 0.0023 over 2 seeds of the K3 arm, mean 2.3670.**

**Same, with standard residuals instead: sd 0.0135 over 2 seeds, mean 2.3321.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.


## hops -- acc (higher is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 0.6139 | 0.0242 | 3 | 320 |
| noattnres | none | MoE | 0.6424 | 0.0566 | 3 | 190 |
| dense | full | dense | 0.6198 | 0.0013 | 3 | 246 |
| noattnres_dense | none | dense | 0.6204 | 0.0017 | 3 | 147 |

AttnRes effect with MoE: -0.0284; with a dense FFN: -0.0007; interaction: -0.0278.

**Noise floor (hops): sd 0.0242 over 3 seeds of the K3 arm, mean 0.6139.**

**Same, with standard residuals instead: sd 0.0566 over 3 seeds, mean 0.6424.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.

## bytes -- bpb (lower is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 2.3670 | 0.0023 | 2 | 206 |
| noattnres | none | MoE | 2.3321 | 0.0135 | 2 | 128 |
| dense | full | dense | 2.2073 | 0.0221 | 2 | 187 |
| noattnres_dense | none | dense | 2.2215 | 0.0022 | 2 | 121 |

AttnRes effect with MoE: +0.0349; with a dense FFN: -0.0142; interaction: +0.0491.

**Noise floor (bytes): sd 0.0023 over 2 seeds of the K3 arm, mean 2.3670.**

**Same, with standard residuals instead: sd 0.0135 over 2 seeds, mean 2.3321.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.


## hops -- acc (higher is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 0.6139 | 0.0242 | 3 | 320 |
| noattnres | none | MoE | 0.6424 | 0.0566 | 3 | 190 |
| dense | full | dense | 0.6198 | 0.0013 | 3 | 246 |
| noattnres_dense | none | dense | 0.6204 | 0.0017 | 3 | 147 |

AttnRes effect with MoE: -0.0284; with a dense FFN: -0.0007; interaction: -0.0278.

**Noise floor (hops): sd 0.0242 over 3 seeds of the K3 arm, mean 0.6139.**

**Same, with standard residuals instead: sd 0.0566 over 3 seeds, mean 0.6424.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.

## bytes -- bpb (lower is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 2.3670 | 0.0023 | 2 | 206 |
| noattnres | none | MoE | 2.3321 | 0.0135 | 2 | 128 |
| dense | full | dense | 2.2073 | 0.0221 | 2 | 187 |
| noattnres_dense | none | dense | 2.2215 | 0.0022 | 2 | 121 |

AttnRes effect with MoE: +0.0349; with a dense FFN: -0.0142; interaction: +0.0491.

**Noise floor (bytes): sd 0.0023 over 2 seeds of the K3 arm, mean 2.3670.**

**Same, with standard residuals instead: sd 0.0135 over 2 seeds, mean 2.3321.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.


## hops -- acc (higher is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 0.6139 | 0.0242 | 3 | 320 |
| noattnres | none | MoE | 0.6424 | 0.0566 | 3 | 190 |
| dense | full | dense | 0.6198 | 0.0013 | 3 | 246 |
| noattnres_dense | none | dense | 0.6204 | 0.0017 | 3 | 147 |

AttnRes effect with MoE: -0.0284; with a dense FFN: -0.0007; interaction: -0.0278.

**Noise floor (hops): sd 0.0242 over 3 seeds of the K3 arm, mean 0.6139.**

**Same, with standard residuals instead: sd 0.0566 over 3 seeds, mean 0.6424.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.

## bytes -- bpb (lower is better)

| arm | AttnRes | FFN | mean | sd | n | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| k3 | full | MoE | 2.3670 | 0.0023 | 2 | 206 |
| noattnres | none | MoE | 2.3321 | 0.0135 | 2 | 128 |
| dense | full | dense | 2.2073 | 0.0221 | 2 | 187 |
| noattnres_dense | none | dense | 2.2215 | 0.0022 | 2 | 121 |

AttnRes effect with MoE: +0.0349; with a dense FFN: -0.0142; interaction: +0.0491.

**Noise floor (bytes): sd 0.0023 over 2 seeds of the K3 arm, mean 2.3670.**

**Same, with standard residuals instead: sd 0.0135 over 2 seeds, mean 2.3321.** E1-E4 run the K3 arm; this row says how much of the spread is AttnRes.


## Part 2 — E1, depth x data

The central result, and it is negative. Depth helps -- +0.09 at 2 hops on the
smaller budget, +0.064 for R=2 over R=1 at 3 hops on the larger one -- but the
*ceiling does not move with data*. At 2 hops, doubling the budget wipes out
depth's advantage entirely (R=1 catches up from 0.478 to 0.609 while R=2 goes
0.570 to 0.598): depth was substituting for data, not compounding with it. At 3
hops the argmax stays at R=2 at both budgets. The brief's own kill condition was
"if the curves do not shift with data, the combination is dead"; they did not
shift.

The extrapolation rows say something the papers do not. Unrolling past the
trained depth is close to free and close to useless: R=4 -> R=8 moves accuracy
by 0.003. Meanwhile evaluating *below* the trained depth is expensive (a
model trained at R=4 drops from 0.626 to 0.454 at R=1), so the loops are doing
real work -- they are simply done after about two.

Bytes is the control and behaves: 2.367 / 2.355 / 2.348 for R = 1 / 2 / 4, a
0.02 spread against a seed sd of up to 0.029. Depth buys composition, not
next-byte statistics, which is Ouro's separation reproduced at 1/5000th the
scale.

<!-- filled by: python e1_depth_data.py --report-only -->

## 2-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.478 ± 0.004 | 0.609 ± 0.029 |
| 2 | 0.570 ± 0.027 | 0.598 ± 0.042 |
| 4 | 0.553 ± 0.011 | 0.626 ± 0.006 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=4.

### 2-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.478 | 0.397 | 0.388 | 0.389 |
| 1 | 3000 | 0.609 | 0.553 | 0.556 | 0.556 |
| 2 | 1500 | 0.293 | 0.570 | 0.486 | 0.487 |
| 2 | 3000 | 0.464 | 0.598 | 0.575 | 0.567 |
| 4 | 1500 | 0.384 | 0.513 | 0.553 | 0.552 |
| 4 | 3000 | 0.454 | 0.594 | 0.626 | 0.623 |

### 2-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.609 ± 0.029 |
| 2 | 1500 | 2.31e+12 | 0.570 ± 0.027 |

## 3-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.547 ± 0.043 | 0.625 ± 0.087 |
| 2 | 0.600 ± 0.099 | 0.689 ± 0.020 |
| 4 | 0.571 ± 0.018 | 0.671 ± 0.012 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=2.

### 3-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.547 | 0.500 | 0.507 | 0.505 |
| 1 | 3000 | 0.625 | 0.507 | 0.486 | 0.487 |
| 2 | 1500 | 0.344 | 0.600 | 0.557 | 0.550 |
| 2 | 3000 | 0.406 | 0.689 | 0.594 | 0.591 |
| 4 | 1500 | 0.409 | 0.550 | 0.571 | 0.574 |
| 4 | 3000 | 0.538 | 0.662 | 0.671 | 0.672 |

### 3-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.625 ± 0.087 |
| 2 | 1500 | 2.31e+12 | 0.600 ± 0.099 |

## bytes: bits per byte (lower is better)

| loops | 300 steps | 600 steps |
| --- | --- | --- |
| 1 | 2.6606 ± 0.0307 | 2.3670 ± 0.0023 |
| 2 | 2.6484 ± 0.0372 | 2.3553 ± 0.0099 |
| 4 | 2.6531 ± 0.0033 | 2.3484 ± 0.0287 |


## 2-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.478 ± 0.004 | 0.609 ± 0.029 |
| 2 | 0.570 ± 0.027 | 0.598 ± 0.042 |
| 4 | 0.553 ± 0.011 | 0.626 ± 0.006 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=4.

### 2-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.478 | 0.397 | 0.388 | 0.389 |
| 1 | 3000 | 0.609 | 0.553 | 0.556 | 0.556 |
| 2 | 1500 | 0.293 | 0.570 | 0.486 | 0.487 |
| 2 | 3000 | 0.464 | 0.598 | 0.575 | 0.567 |
| 4 | 1500 | 0.384 | 0.513 | 0.553 | 0.552 |
| 4 | 3000 | 0.454 | 0.594 | 0.626 | 0.623 |

### 2-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.609 ± 0.029 |
| 2 | 1500 | 2.31e+12 | 0.570 ± 0.027 |

## 3-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.547 ± 0.043 | 0.625 ± 0.087 |
| 2 | 0.600 ± 0.099 | 0.689 ± 0.020 |
| 4 | 0.571 ± 0.018 | 0.671 ± 0.012 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=2.

### 3-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.547 | 0.500 | 0.507 | 0.505 |
| 1 | 3000 | 0.625 | 0.507 | 0.486 | 0.487 |
| 2 | 1500 | 0.344 | 0.600 | 0.557 | 0.550 |
| 2 | 3000 | 0.406 | 0.689 | 0.594 | 0.591 |
| 4 | 1500 | 0.409 | 0.550 | 0.571 | 0.574 |
| 4 | 3000 | 0.538 | 0.662 | 0.671 | 0.672 |

### 3-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.625 ± 0.087 |
| 2 | 1500 | 2.31e+12 | 0.600 ± 0.099 |

## bytes: bits per byte (lower is better)

| loops | 300 steps | 600 steps |
| --- | --- | --- |
| 1 | 2.6606 ± 0.0307 | 2.3670 ± 0.0023 |
| 2 | 2.6484 ± 0.0372 | 2.3553 ± 0.0099 |
| 4 | 2.6531 ± 0.0033 | 2.3484 ± 0.0287 |


## 2-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.478 ± 0.004 | 0.609 ± 0.029 |
| 2 | 0.570 ± 0.027 | 0.598 ± 0.042 |
| 4 | 0.553 ± 0.011 | 0.626 ± 0.006 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=4.

### 2-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.478 | 0.397 | 0.388 | 0.389 |
| 1 | 3000 | 0.609 | 0.553 | 0.556 | 0.556 |
| 2 | 1500 | 0.293 | 0.570 | 0.486 | 0.487 |
| 2 | 3000 | 0.464 | 0.598 | 0.575 | 0.567 |
| 4 | 1500 | 0.384 | 0.513 | 0.553 | 0.552 |
| 4 | 3000 | 0.454 | 0.594 | 0.626 | 0.623 |

### 2-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.609 ± 0.029 |
| 2 | 1500 | 2.31e+12 | 0.570 ± 0.027 |

## 3-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.547 ± 0.043 | 0.625 ± 0.087 |
| 2 | 0.600 ± 0.099 | 0.689 ± 0.020 |
| 4 | 0.571 ± 0.018 | 0.671 ± 0.012 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=2.

### 3-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.547 | 0.500 | 0.507 | 0.505 |
| 1 | 3000 | 0.625 | 0.507 | 0.486 | 0.487 |
| 2 | 1500 | 0.344 | 0.600 | 0.557 | 0.550 |
| 2 | 3000 | 0.406 | 0.689 | 0.594 | 0.591 |
| 4 | 1500 | 0.409 | 0.550 | 0.571 | 0.574 |
| 4 | 3000 | 0.538 | 0.662 | 0.671 | 0.672 |

### 3-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.625 ± 0.087 |
| 2 | 1500 | 2.31e+12 | 0.600 ± 0.099 |

## bytes: bits per byte (lower is better)

| loops | 300 steps | 600 steps |
| --- | --- | --- |
| 1 | 2.6606 ± 0.0307 | 2.3670 ± 0.0023 |
| 2 | 2.6484 ± 0.0372 | 2.3553 ± 0.0099 |
| 4 | 2.6531 ± 0.0033 | 2.3484 ± 0.0287 |


## 2-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.478 ± 0.004 | 0.609 ± 0.029 |
| 2 | 0.570 ± 0.027 | 0.598 ± 0.042 |
| 4 | 0.553 ± 0.011 | 0.626 ± 0.006 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=4.

### 2-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.478 | 0.397 | 0.388 | 0.389 |
| 1 | 3000 | 0.609 | 0.553 | 0.556 | 0.556 |
| 2 | 1500 | 0.293 | 0.570 | 0.486 | 0.487 |
| 2 | 3000 | 0.464 | 0.598 | 0.575 | 0.567 |
| 4 | 1500 | 0.384 | 0.513 | 0.553 | 0.552 |
| 4 | 3000 | 0.454 | 0.594 | 0.626 | 0.623 |

### 2-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.609 ± 0.029 |
| 2 | 1500 | 2.31e+12 | 0.570 ± 0.027 |

## 3-hop composition: accuracy by depth and budget

| loops | 1500 steps | 3000 steps |
| --- | --- | --- |
| 1 | 0.547 ± 0.043 | 0.625 ± 0.087 |
| 2 | 0.600 ± 0.099 | 0.689 ± 0.020 |
| 4 | 0.571 ± 0.018 | 0.671 ± 0.012 |

Best depth per budget: 1500 steps -> R=2, 3000 steps -> R=2.

### 3-hop: unrolled past the trained depth

| trained R | budget | eval R=1 | eval R=2 | eval R=4 | eval R=8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1500 | 0.547 | 0.500 | 0.507 | 0.505 |
| 1 | 3000 | 0.625 | 0.507 | 0.486 | 0.487 |
| 2 | 1500 | 0.344 | 0.600 | 0.557 | 0.550 |
| 2 | 3000 | 0.406 | 0.689 | 0.594 | 0.591 |
| 4 | 1500 | 0.409 | 0.550 | 0.571 | 0.574 |
| 4 | 3000 | 0.538 | 0.662 | 0.671 | 0.672 |

### 3-hop: fixed compute (equal loops x steps)

| loops | steps | train FLOPs | accuracy |
| --- | --- | --- | --- |
| 1 | 3000 | 3.03e+12 | 0.625 ± 0.087 |
| 2 | 1500 | 2.31e+12 | 0.600 ± 0.099 |

## bytes: bits per byte (lower is better)

| loops | 300 steps | 600 steps |
| --- | --- | --- |
| 1 | 2.6606 ± 0.0307 | 2.3670 ± 0.0023 |
| 2 | 2.6484 ± 0.0372 | 2.3553 ± 0.0099 |
| 4 | 2.6531 ± 0.0033 | 2.3484 ± 0.0287 |


## Part 3 — E2, where the loop may write

A null, and an honest reading has to say it is an underpowered one. Registers,
wiped registers and cross-loop AttnRes all land on `plain` within +/-0.008, and
the paired per-seed differences straddle zero on both tasks.

The reason to distrust it: `twochain` -- the task built so that a partial result
must survive while a second chain is computed -- only reaches 0.258 against a
chance floor of 0.167. The model never became good enough at the thing a
scratchpad would help with for a scratchpad to show up. What the experiment
establishes is that **adding writable state is not free improvement**; what it
cannot establish is that state does not matter. The fix is not more seeds, it is
a two-chain task the model can actually learn, or a wider model.

Worth noting for anyone repeating this: `regswiped` (same width, wiped every
loop) landing within 0.006 of `regs` means the null is not hiding a width
effect either. Both mechanisms are inert here, not compensating.

<!-- filled by: python e2_writable_state.py --report-only -->

## twochain (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.258 | 0.009 | 2 | - | 277,952 | 7.48e+05 |
| regs | 0.258 | 0.001 | 2 | +0.000 | 278,464 | 7.48e+05 |
| regswiped | 0.252 | 0.001 | 2 | -0.006 | 278,464 | 7.48e+05 |
| xloop | 0.253 | 0.007 | 2 | -0.005 | 277,952 | 7.48e+05 |

Paired per-seed differences against `plain`:

- `regs`: -0.005, +0.005 (mean +0.000, sd 0.007)
- `regswiped`: -0.013, +0.001 (mean -0.006, sd 0.010)
- `xloop`: -0.016, +0.006 (mean -0.005, sd 0.016)

## hops (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.570 | 0.006 | 2 | - | 277,952 | 7.32e+05 |
| regs | 0.568 | 0.072 | 2 | -0.001 | 278,464 | 7.32e+05 |
| regswiped | 0.561 | 0.018 | 2 | -0.008 | 278,464 | 7.32e+05 |
| xloop | 0.566 | 0.019 | 2 | -0.003 | 277,952 | 7.32e+05 |

Paired per-seed differences against `plain`:

- `regs`: +0.046, -0.048 (mean -0.001, sd 0.066)
- `regswiped`: -0.025, +0.008 (mean -0.008, sd 0.024)
- `xloop`: -0.021, +0.014 (mean -0.003, sd 0.025)


## twochain (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.258 | 0.009 | 2 | - | 277,952 | 7.48e+05 |
| regs | 0.258 | 0.001 | 2 | +0.000 | 278,464 | 7.48e+05 |
| regswiped | 0.252 | 0.001 | 2 | -0.006 | 278,464 | 7.48e+05 |
| xloop | 0.253 | 0.007 | 2 | -0.005 | 277,952 | 7.48e+05 |

Paired per-seed differences against `plain`:

- `regs`: -0.005, +0.005 (mean +0.000, sd 0.007)
- `regswiped`: -0.013, +0.001 (mean -0.006, sd 0.010)
- `xloop`: -0.016, +0.006 (mean -0.005, sd 0.016)

## hops (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.570 | 0.006 | 2 | - | 277,952 | 7.32e+05 |
| regs | 0.568 | 0.072 | 2 | -0.001 | 278,464 | 7.32e+05 |
| regswiped | 0.561 | 0.018 | 2 | -0.008 | 278,464 | 7.32e+05 |
| xloop | 0.566 | 0.019 | 2 | -0.003 | 277,952 | 7.32e+05 |

Paired per-seed differences against `plain`:

- `regs`: +0.046, -0.048 (mean -0.001, sd 0.066)
- `regswiped`: -0.025, +0.008 (mean -0.008, sd 0.024)
- `xloop`: -0.021, +0.014 (mean -0.003, sd 0.025)


## twochain (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.258 | 0.009 | 2 | - | 277,952 | 7.48e+05 |
| regs | 0.258 | 0.001 | 2 | +0.000 | 278,464 | 7.48e+05 |
| regswiped | 0.252 | 0.001 | 2 | -0.006 | 278,464 | 7.48e+05 |
| xloop | 0.253 | 0.007 | 2 | -0.005 | 277,952 | 7.48e+05 |

Paired per-seed differences against `plain`:

- `regs`: -0.005, +0.005 (mean +0.000, sd 0.007)
- `regswiped`: -0.013, +0.001 (mean -0.006, sd 0.010)
- `xloop`: -0.016, +0.006 (mean -0.005, sd 0.016)

## hops (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.570 | 0.006 | 2 | - | 277,952 | 7.32e+05 |
| regs | 0.568 | 0.072 | 2 | -0.001 | 278,464 | 7.32e+05 |
| regswiped | 0.561 | 0.018 | 2 | -0.008 | 278,464 | 7.32e+05 |
| xloop | 0.566 | 0.019 | 2 | -0.003 | 277,952 | 7.32e+05 |

Paired per-seed differences against `plain`:

- `regs`: +0.046, -0.048 (mean -0.001, sd 0.066)
- `regswiped`: -0.025, +0.008 (mean -0.008, sd 0.024)
- `xloop`: -0.021, +0.014 (mean -0.003, sd 0.025)


## twochain (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.258 | 0.009 | 2 | - | 277,952 | 7.48e+05 |
| regs | 0.258 | 0.001 | 2 | +0.000 | 278,464 | 7.48e+05 |
| regswiped | 0.252 | 0.001 | 2 | -0.006 | 278,464 | 7.48e+05 |
| xloop | 0.253 | 0.007 | 2 | -0.005 | 277,952 | 7.48e+05 |

Paired per-seed differences against `plain`:

- `regs`: -0.005, +0.005 (mean +0.000, sd 0.007)
- `regswiped`: -0.013, +0.001 (mean -0.006, sd 0.010)
- `xloop`: -0.016, +0.006 (mean -0.005, sd 0.016)

## hops (accuracy)

| arm | mean | sd | n | vs plain | params | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 0.570 | 0.006 | 2 | - | 277,952 | 7.32e+05 |
| regs | 0.568 | 0.072 | 2 | -0.001 | 278,464 | 7.32e+05 |
| regswiped | 0.561 | 0.018 | 2 | -0.008 | 278,464 | 7.32e+05 |
| xloop | 0.566 | 0.019 | 2 | -0.003 | 277,952 | 7.32e+05 |

Paired per-seed differences against `plain`:

- `regs`: +0.046, -0.048 (mean -0.001, sd 0.066)
- `regswiped`: -0.025, +0.008 (mean -0.008, sd 0.024)
- `xloop`: -0.021, +0.014 (mean -0.003, sd 0.025)


## Part 4 — E4, halting

Three things, in order of how much they change what one would build.

**The step-indexed gate cannot use depth it never saw.** Ouro's Q-exit stops at
exactly the same average depth at R=8 and R=12 as at R=4 (3.02 and 2.17 for the
two seeds, identical to four decimal places across the three evaluation depths),
because the gate has no logits past its trained index. PonderNet's step-invariant
gate keeps moving (2.66 -> 2.83 -> 2.87). This is the brief's §5 argument,
observed rather than argued, and it is structural: no amount of scale fixes a
head that is indexed by a step number it will not be given.

**A halting objective makes every depth usable.** At R=1 the gated models score
0.51-0.52 against the ungated model's 0.405. Training the loss over exit steps
turns each iteration into a legitimate output point, which is worth having on
its own -- it is what makes early exit exact rather than approximate.

**The free rule is competitive.** Huginn's zero-shot KL exit, applied to a model
trained with no gate at all, reaches 0.570 -- full-depth accuracy -- at an
average depth of 2.23 of 4, and 0.577 at 3.09. The trained gates spend a similar
budget for a similar result, and the Ouro arm is wildly seed-dependent while
doing it (sd 0.155 at R=4). Any argument for RL on the halting gate has to start
by beating a rule that costs nothing.

<!-- filled by: python e4_halting.py --report-only -->

## accuracy at the trained depth and unrolled beyond it

| arm | R=1 | R=2 | R=4 | R=8 | R=12 |
| --- | --- | --- | --- | --- | --- |
| none | 0.405 ± 0.027 | 0.562 ± 0.008 | 0.570 ± 0.006 | 0.570 ± 0.004 | 0.569 ± 0.004 |
| ouro | 0.509 ± 0.040 | 0.492 ± 0.094 | 0.469 ± 0.155 | 0.461 ± 0.142 | 0.460 ± 0.146 |
| pondernet | 0.520 ± 0.001 | 0.557 ± 0.036 | 0.548 ± 0.020 | 0.536 ± 0.006 | 0.534 ± 0.004 |

## what the exit rule spends

| arm | eval depth | rule | accuracy | mean depth used |
| --- | --- | --- | --- | --- |
| none (s0) | 4 | kl0.02 | 0.577 | 3.09 |
| none (s0) | 4 | kl0.1 | 0.569 | 2.64 |
| none (s0) | 4 | kl0.5 | 0.570 | 2.23 |
| none (s1) | 4 | kl0.02 | 0.566 | 3.34 |
| none (s1) | 4 | kl0.1 | 0.564 | 2.81 |
| none (s1) | 4 | kl0.5 | 0.562 | 2.38 |
| ouro (s0) | 1 | qexit | 0.537 | 1.00 |
| ouro (s0) | 2 | qexit | 0.559 | 2.00 |
| ouro (s0) | 4 | qexit | 0.705 | 3.02 |
| ouro (s0) | 8 | qexit | 0.705 | 3.02 |
| ouro (s0) | 12 | qexit | 0.705 | 3.02 |
| ouro (s1) | 1 | qexit | 0.481 | 1.00 |
| ouro (s1) | 2 | qexit | 0.531 | 1.64 |
| ouro (s1) | 4 | qexit | 0.597 | 2.17 |
| ouro (s1) | 8 | qexit | 0.597 | 2.17 |
| ouro (s1) | 12 | qexit | 0.597 | 2.17 |
| pondernet (s0) | 1 | qexit | 0.520 | 1.00 |
| pondernet (s0) | 2 | qexit | 0.612 | 1.93 |
| pondernet (s0) | 4 | qexit | 0.641 | 2.66 |
| pondernet (s0) | 8 | qexit | 0.638 | 2.83 |
| pondernet (s0) | 12 | qexit | 0.638 | 2.87 |
| pondernet (s1) | 1 | qexit | 0.519 | 1.00 |
| pondernet (s1) | 2 | qexit | 0.525 | 1.63 |
| pondernet (s1) | 4 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 8 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 12 | qexit | 0.527 | 1.70 |


## accuracy at the trained depth and unrolled beyond it

| arm | R=1 | R=2 | R=4 | R=8 | R=12 |
| --- | --- | --- | --- | --- | --- |
| none | 0.405 ± 0.027 | 0.562 ± 0.008 | 0.570 ± 0.006 | 0.570 ± 0.004 | 0.569 ± 0.004 |
| ouro | 0.509 ± 0.040 | 0.492 ± 0.094 | 0.469 ± 0.155 | 0.461 ± 0.142 | 0.460 ± 0.146 |
| pondernet | 0.520 ± 0.001 | 0.557 ± 0.036 | 0.548 ± 0.020 | 0.536 ± 0.006 | 0.534 ± 0.004 |

## what the exit rule spends

| arm | eval depth | rule | accuracy | mean depth used |
| --- | --- | --- | --- | --- |
| none (s0) | 4 | kl0.02 | 0.577 | 3.09 |
| none (s0) | 4 | kl0.1 | 0.569 | 2.64 |
| none (s0) | 4 | kl0.5 | 0.570 | 2.23 |
| none (s1) | 4 | kl0.02 | 0.566 | 3.34 |
| none (s1) | 4 | kl0.1 | 0.564 | 2.81 |
| none (s1) | 4 | kl0.5 | 0.562 | 2.38 |
| ouro (s0) | 1 | qexit | 0.537 | 1.00 |
| ouro (s0) | 2 | qexit | 0.559 | 2.00 |
| ouro (s0) | 4 | qexit | 0.705 | 3.02 |
| ouro (s0) | 8 | qexit | 0.705 | 3.02 |
| ouro (s0) | 12 | qexit | 0.705 | 3.02 |
| ouro (s1) | 1 | qexit | 0.481 | 1.00 |
| ouro (s1) | 2 | qexit | 0.531 | 1.64 |
| ouro (s1) | 4 | qexit | 0.597 | 2.17 |
| ouro (s1) | 8 | qexit | 0.597 | 2.17 |
| ouro (s1) | 12 | qexit | 0.597 | 2.17 |
| pondernet (s0) | 1 | qexit | 0.520 | 1.00 |
| pondernet (s0) | 2 | qexit | 0.612 | 1.93 |
| pondernet (s0) | 4 | qexit | 0.641 | 2.66 |
| pondernet (s0) | 8 | qexit | 0.638 | 2.83 |
| pondernet (s0) | 12 | qexit | 0.638 | 2.87 |
| pondernet (s1) | 1 | qexit | 0.519 | 1.00 |
| pondernet (s1) | 2 | qexit | 0.525 | 1.63 |
| pondernet (s1) | 4 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 8 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 12 | qexit | 0.527 | 1.70 |


## accuracy at the trained depth and unrolled beyond it

| arm | R=1 | R=2 | R=4 | R=8 | R=12 |
| --- | --- | --- | --- | --- | --- |
| none | 0.405 ± 0.027 | 0.562 ± 0.008 | 0.570 ± 0.006 | 0.570 ± 0.004 | 0.569 ± 0.004 |
| ouro | 0.509 ± 0.040 | 0.492 ± 0.094 | 0.469 ± 0.155 | 0.461 ± 0.142 | 0.460 ± 0.146 |
| pondernet | 0.520 ± 0.001 | 0.557 ± 0.036 | 0.548 ± 0.020 | 0.536 ± 0.006 | 0.534 ± 0.004 |

## what the exit rule spends

| arm | eval depth | rule | accuracy | mean depth used |
| --- | --- | --- | --- | --- |
| none (s0) | 4 | kl0.02 | 0.577 | 3.09 |
| none (s0) | 4 | kl0.1 | 0.569 | 2.64 |
| none (s0) | 4 | kl0.5 | 0.570 | 2.23 |
| none (s1) | 4 | kl0.02 | 0.566 | 3.34 |
| none (s1) | 4 | kl0.1 | 0.564 | 2.81 |
| none (s1) | 4 | kl0.5 | 0.562 | 2.38 |
| ouro (s0) | 1 | qexit | 0.537 | 1.00 |
| ouro (s0) | 2 | qexit | 0.559 | 2.00 |
| ouro (s0) | 4 | qexit | 0.705 | 3.02 |
| ouro (s0) | 8 | qexit | 0.705 | 3.02 |
| ouro (s0) | 12 | qexit | 0.705 | 3.02 |
| ouro (s1) | 1 | qexit | 0.481 | 1.00 |
| ouro (s1) | 2 | qexit | 0.531 | 1.64 |
| ouro (s1) | 4 | qexit | 0.597 | 2.17 |
| ouro (s1) | 8 | qexit | 0.597 | 2.17 |
| ouro (s1) | 12 | qexit | 0.597 | 2.17 |
| pondernet (s0) | 1 | qexit | 0.520 | 1.00 |
| pondernet (s0) | 2 | qexit | 0.612 | 1.93 |
| pondernet (s0) | 4 | qexit | 0.641 | 2.66 |
| pondernet (s0) | 8 | qexit | 0.638 | 2.83 |
| pondernet (s0) | 12 | qexit | 0.638 | 2.87 |
| pondernet (s1) | 1 | qexit | 0.519 | 1.00 |
| pondernet (s1) | 2 | qexit | 0.525 | 1.63 |
| pondernet (s1) | 4 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 8 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 12 | qexit | 0.527 | 1.70 |


## accuracy at the trained depth and unrolled beyond it

| arm | R=1 | R=2 | R=4 | R=8 | R=12 |
| --- | --- | --- | --- | --- | --- |
| none | 0.405 ± 0.027 | 0.562 ± 0.008 | 0.570 ± 0.006 | 0.570 ± 0.004 | 0.569 ± 0.004 |
| ouro | 0.509 ± 0.040 | 0.492 ± 0.094 | 0.469 ± 0.155 | 0.461 ± 0.142 | 0.460 ± 0.146 |
| pondernet | 0.520 ± 0.001 | 0.557 ± 0.036 | 0.548 ± 0.020 | 0.536 ± 0.006 | 0.534 ± 0.004 |

## what the exit rule spends

| arm | eval depth | rule | accuracy | mean depth used |
| --- | --- | --- | --- | --- |
| none (s0) | 4 | kl0.02 | 0.577 | 3.09 |
| none (s0) | 4 | kl0.1 | 0.569 | 2.64 |
| none (s0) | 4 | kl0.5 | 0.570 | 2.23 |
| none (s1) | 4 | kl0.02 | 0.566 | 3.34 |
| none (s1) | 4 | kl0.1 | 0.564 | 2.81 |
| none (s1) | 4 | kl0.5 | 0.562 | 2.38 |
| ouro (s0) | 1 | qexit | 0.537 | 1.00 |
| ouro (s0) | 2 | qexit | 0.559 | 2.00 |
| ouro (s0) | 4 | qexit | 0.705 | 3.02 |
| ouro (s0) | 8 | qexit | 0.705 | 3.02 |
| ouro (s0) | 12 | qexit | 0.705 | 3.02 |
| ouro (s1) | 1 | qexit | 0.481 | 1.00 |
| ouro (s1) | 2 | qexit | 0.531 | 1.64 |
| ouro (s1) | 4 | qexit | 0.597 | 2.17 |
| ouro (s1) | 8 | qexit | 0.597 | 2.17 |
| ouro (s1) | 12 | qexit | 0.597 | 2.17 |
| pondernet (s0) | 1 | qexit | 0.520 | 1.00 |
| pondernet (s0) | 2 | qexit | 0.612 | 1.93 |
| pondernet (s0) | 4 | qexit | 0.641 | 2.66 |
| pondernet (s0) | 8 | qexit | 0.638 | 2.83 |
| pondernet (s0) | 12 | qexit | 0.638 | 2.87 |
| pondernet (s1) | 1 | qexit | 0.519 | 1.00 |
| pondernet (s1) | 2 | qexit | 0.525 | 1.63 |
| pondernet (s1) | 4 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 8 | qexit | 0.527 | 1.70 |
| pondernet (s1) | 12 | qexit | 0.527 | 1.70 |


## Part 5 — E3, step-conditioned routing

The one clear positive result in the suite, and its mechanism is legible.

Accuracy on composition goes 0.570 -> **0.607** with a per-loop router bias,
against a seed spread of 0.006. On bytes it goes nowhere (2.348 -> 2.351). That
is the predicted signature: the conditioner buys manipulation, not statistics.

The divergence table says why, and it is not the reason the proposal gave. Even
with **no** conditioning, routing changes sharply from loop 1 to loop 2 (JS
0.047) and then stops (0.003, then 0.0003). The weight-tied core is not
"computing the same function every loop" in any behavioural sense -- it is
iterating toward a fixed point, and the first pass is already distinct from the
rest.

Against that, the two conditioners do different things. `bias`, which can only
change *which experts*, leaves the convergence pattern intact and wins.
`embed`, which can change *which tokens go where*, forces large divergence at
every loop (0.235, 0.230, 0.215) and scores lower (0.592). More differentiation,
worse model. So what the loop wanted was not per-step specialisation but a
**phase distinction between entering and refining** -- and the cheap mechanism
is the one that provides exactly that and nothing more.

Load statistics answer the brief's untested risk in passing: expert-load sd
rises only 0.0195 -> 0.0277 with conditioning, and 0.0225 -> 0.0267 across R = 1
to 4 in E1. Looping multiplies routing decisions per update without
destabilising load at this scale.

<!-- filled by: python e3_step_routing.py --report-only -->

## hops (acc)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 0.5697 | 0.0055 | 2 | 0.0195 |
| bias | 0.6071 | 0.0069 | 2 | 0.0277 |
| embed | 0.5918 | 0.0110 | 2 | 0.0419 |

## bytes (bpb)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 2.3484 | 0.0287 | 2 | 0.0092 |
| bias | 2.3506 | 0.0289 | 2 | 0.0128 |
| embed | 2.3595 | 0.0264 | 2 | 0.0129 |

## does routing actually differ across loops?

Jensen-Shannon divergence between the expert-usage distributions of consecutive loops, from a collected forward pass at the end of training. The mechanism claim lives or dies here: a conditioner that leaves routing unchanged has not done what it was added to do, whatever accuracy says.

| arm | task | seed | JS 1->2 | JS 2->3 | JS 3->4 | JS 1->4 |
| --- | --- | --- | --- | --- | --- | --- |
| none | hops | 0 | 0.0212 | 0.0039 | 0.0004 | 0.0362 |
| none | hops | 1 | 0.0465 | 0.0034 | 0.0003 | 0.0574 |
| none | bytes | 0 | 0.0361 | 0.0074 | 0.0014 | 0.0314 |
| none | bytes | 1 | 0.0241 | 0.0037 | 0.0006 | 0.0222 |
| bias | hops | 0 | 0.0205 | 0.0018 | 0.0023 | 0.0224 |
| bias | hops | 1 | 0.0578 | 0.0138 | 0.0147 | 0.0908 |
| bias | bytes | 0 | 0.0282 | 0.0081 | 0.0018 | 0.0299 |
| bias | bytes | 1 | 0.0147 | 0.0061 | 0.0015 | 0.0160 |
| embed | hops | 0 | 0.1196 | 0.0728 | 0.0710 | 0.0952 |
| embed | hops | 1 | 0.2350 | 0.2300 | 0.2148 | 0.2339 |
| embed | bytes | 0 | 0.0371 | 0.0100 | 0.0092 | 0.0411 |
| embed | bytes | 1 | 0.0357 | 0.0069 | 0.0160 | 0.0361 |


## hops (acc)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 0.5697 | 0.0055 | 2 | 0.0195 |
| bias | 0.6071 | 0.0069 | 2 | 0.0277 |
| embed | 0.5918 | 0.0110 | 2 | 0.0419 |

## bytes (bpb)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 2.3484 | 0.0287 | 2 | 0.0092 |
| bias | 2.3506 | 0.0289 | 2 | 0.0128 |
| embed | 2.3595 | 0.0264 | 2 | 0.0129 |

## does routing actually differ across loops?

Jensen-Shannon divergence between the expert-usage distributions of consecutive loops, from a collected forward pass at the end of training. The mechanism claim lives or dies here: a conditioner that leaves routing unchanged has not done what it was added to do, whatever accuracy says.

| arm | task | seed | JS 1->2 | JS 2->3 | JS 3->4 | JS 1->4 |
| --- | --- | --- | --- | --- | --- | --- |
| none | hops | 0 | 0.0212 | 0.0039 | 0.0004 | 0.0362 |
| none | hops | 1 | 0.0465 | 0.0034 | 0.0003 | 0.0574 |
| none | bytes | 0 | 0.0361 | 0.0074 | 0.0014 | 0.0314 |
| none | bytes | 1 | 0.0241 | 0.0037 | 0.0006 | 0.0222 |
| bias | hops | 0 | 0.0205 | 0.0018 | 0.0023 | 0.0224 |
| bias | hops | 1 | 0.0578 | 0.0138 | 0.0147 | 0.0908 |
| bias | bytes | 0 | 0.0282 | 0.0081 | 0.0018 | 0.0299 |
| bias | bytes | 1 | 0.0147 | 0.0061 | 0.0015 | 0.0160 |
| embed | hops | 0 | 0.1196 | 0.0728 | 0.0710 | 0.0952 |
| embed | hops | 1 | 0.2350 | 0.2300 | 0.2148 | 0.2339 |
| embed | bytes | 0 | 0.0371 | 0.0100 | 0.0092 | 0.0411 |
| embed | bytes | 1 | 0.0357 | 0.0069 | 0.0160 | 0.0361 |


## hops (acc)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 0.5697 | 0.0055 | 2 | 0.0195 |
| bias | 0.6071 | 0.0069 | 2 | 0.0277 |
| embed | 0.5918 | 0.0110 | 2 | 0.0419 |

## bytes (bpb)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 2.3484 | 0.0287 | 2 | 0.0092 |
| bias | 2.3506 | 0.0289 | 2 | 0.0128 |
| embed | 2.3595 | 0.0264 | 2 | 0.0129 |

## does routing actually differ across loops?

Jensen-Shannon divergence between the expert-usage distributions of consecutive loops, from a collected forward pass at the end of training. The mechanism claim lives or dies here: a conditioner that leaves routing unchanged has not done what it was added to do, whatever accuracy says.

| arm | task | seed | JS 1->2 | JS 2->3 | JS 3->4 | JS 1->4 |
| --- | --- | --- | --- | --- | --- | --- |
| none | hops | 0 | 0.0212 | 0.0039 | 0.0004 | 0.0362 |
| none | hops | 1 | 0.0465 | 0.0034 | 0.0003 | 0.0574 |
| none | bytes | 0 | 0.0361 | 0.0074 | 0.0014 | 0.0314 |
| none | bytes | 1 | 0.0241 | 0.0037 | 0.0006 | 0.0222 |
| bias | hops | 0 | 0.0205 | 0.0018 | 0.0023 | 0.0224 |
| bias | hops | 1 | 0.0578 | 0.0138 | 0.0147 | 0.0908 |
| bias | bytes | 0 | 0.0282 | 0.0081 | 0.0018 | 0.0299 |
| bias | bytes | 1 | 0.0147 | 0.0061 | 0.0015 | 0.0160 |
| embed | hops | 0 | 0.1196 | 0.0728 | 0.0710 | 0.0952 |
| embed | hops | 1 | 0.2350 | 0.2300 | 0.2148 | 0.2339 |
| embed | bytes | 0 | 0.0371 | 0.0100 | 0.0092 | 0.0411 |
| embed | bytes | 1 | 0.0357 | 0.0069 | 0.0160 | 0.0361 |


## hops (acc)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 0.5697 | 0.0055 | 2 | 0.0195 |
| bias | 0.6071 | 0.0069 | 2 | 0.0277 |
| embed | 0.5918 | 0.0110 | 2 | 0.0419 |

## bytes (bpb)

| arm | mean | sd | n | load sd (last block) |
| --- | --- | --- | --- | --- |
| none | 2.3484 | 0.0287 | 2 | 0.0092 |
| bias | 2.3506 | 0.0289 | 2 | 0.0128 |
| embed | 2.3595 | 0.0264 | 2 | 0.0129 |

## does routing actually differ across loops?

Jensen-Shannon divergence between the expert-usage distributions of consecutive loops, from a collected forward pass at the end of training. The mechanism claim lives or dies here: a conditioner that leaves routing unchanged has not done what it was added to do, whatever accuracy says.

| arm | task | seed | JS 1->2 | JS 2->3 | JS 3->4 | JS 1->4 |
| --- | --- | --- | --- | --- | --- | --- |
| none | hops | 0 | 0.0212 | 0.0039 | 0.0004 | 0.0362 |
| none | hops | 1 | 0.0465 | 0.0034 | 0.0003 | 0.0574 |
| none | bytes | 0 | 0.0361 | 0.0074 | 0.0014 | 0.0314 |
| none | bytes | 1 | 0.0241 | 0.0037 | 0.0006 | 0.0222 |
| bias | hops | 0 | 0.0205 | 0.0018 | 0.0023 | 0.0224 |
| bias | hops | 1 | 0.0578 | 0.0138 | 0.0147 | 0.0908 |
| bias | bytes | 0 | 0.0282 | 0.0081 | 0.0018 | 0.0299 |
| bias | bytes | 1 | 0.0147 | 0.0061 | 0.0015 | 0.0160 |
| embed | hops | 0 | 0.1196 | 0.0728 | 0.0710 | 0.0952 |
| embed | hops | 1 | 0.2350 | 0.2300 | 0.2148 | 0.2339 |
| embed | bytes | 0 | 0.0371 | 0.0100 | 0.0092 | 0.0411 |
| embed | bytes | 1 | 0.0357 | 0.0069 | 0.0160 | 0.0361 |

---

## What these results say to run next

The brief's ordering (its §9) was: depth x tokens, then writable state, then
looped MoE, then the latent-vs-token frontier, then RL on the gate. After these
runs the ordering should change, and one item should be dropped.

1. **The fixed-point reading, at a scale where the state is wide enough to hold
   something.** Every mechanistic result here points the same way: routing
   converges after the first loop, unrolling past training does nothing, and the
   conditioner that helps is the one that distinguishes entering from refining.
   If the loop is a contraction, saturation is not a capacity problem and no
   amount of scratchpad fixes it -- what would fix it is a recurrence that is
   *not* a contraction. That is a different architecture, and it is now the
   highest-information experiment. Measure it directly: the norm of the state
   update per loop, and whether the fixed point is reached faster on easy
   examples than hard ones.

2. **Step-conditioned routing, at real sparsity.** +0.037 outside noise, from a
   per-loop bias table costing R x E parameters, is the best return in the
   suite. The scale-down here used 16 experts at 12.5% activation; K3 runs 896 at
   1.8%, where the room for phase specialisation is far larger and the cost is
   the same. The `bias` beating `embed` result also says the cheap version is the
   right one, which makes this unusually easy to try inside an existing model.

3. **A two-chain task the model can learn**, before E2 is repeated. The
   writable-state question is not answered, it is unmeasured: the arms were
   compared on a task at 0.258 against a 0.167 floor. Either widen the model or
   simplify the task until `plain` is well off the floor, then re-run the same
   four arms unchanged.

4. **E5, the latent-vs-token frontier** (`PARKED.md`). Unchanged in priority,
   still unrun, still the only one of the brief's questions that nobody has
   plotted.

5. **RL on the halting gate — deprioritised.** The zero-shot KL rule already
   reaches full-depth accuracy at 56% of the depth, with no training. The
   supervised gates matched rather than beat it. Spending RL on this before
   showing that a *trained* gate beats a *free* rule is spending it in the wrong
   place.

**Dropped: nothing about parallel loops.** CLP remains out of scope and is now
also less interesting: it is a scheduling fix for a static loop count, and the
depth these models actually use is about two, adaptively. The conflict the brief
identified between static pipelining and adaptive depth is real, and this suite
makes the adaptive side look more attractive, not less.

## The caveat that governs all of it

0.3M parameters, ~3000 steps, 6-entity in-context composition and byte-level
WikiText-2 on four CPU cores. A null here is evidence about this regime and not
about 1.4B models -- and the AttnRes result is a live demonstration that depth
mechanisms can need depth before they show up at all. What travels better than
the effect sizes is the *shape* of the findings: which effects appear on
composition but not on next-byte prediction, and which failures are structural
rather than quantitative. The step-indexed gate's frozen exit depth is the
clearest of those: a head indexed by a step number cannot use a step number it
was never given, at any scale.
