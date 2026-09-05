# Results

Every table in this file is regenerated from `results/*.json` by the
experiment's own `--report-only`. The pilot section is hand-written because the
pilots are the part that failed, and a study that only records the runs that
worked is not recording the thing that decided its design.

---

## Part 0 — the pilots, and three things that were wrong

Recorded because each one changed the experiment, and because the second and
third are mistakes anyone rebuilding this would repeat.

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

### 0.4 What the pilots settled about scale

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

### 0.5 Attention Residuals is a drag at this scale — but not a wall

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

<!-- filled by: python e0_baseline.py --report-only -->

## Part 2 — E1, depth x data

<!-- filled by: python e1_depth_data.py --report-only -->

## Part 3 — E2, where the loop may write

<!-- filled by: python e2_writable_state.py --report-only -->

## Part 4 — E4, halting

<!-- filled by: python e4_halting.py --report-only -->

## Part 5 — E3, step-conditioned routing

<!-- filled by: python e3_step_routing.py --report-only -->
