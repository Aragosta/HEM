# Design: what is being tested, why these arms, and what they cannot show

## 1. Decomposing the brief

The brief (`../docs/` is not where it lives; it is reproduced in `README.md`
§1) proposes a synthesis architecture and then lists six things to run. Taken
as stated the list mixes questions with engineering. Sorted by *what varies*,
the proposals fall into five disjoint axes, and this suite has one experiment
per axis:

| axis | what varies | experiment |
| --- | --- | --- |
| how much depth, against how much data | `loops`, training budget | **E1** |
| what the loop may read and write | `registers`, `loop_memory` | **E2** |
| what the loop computes at each step | `step_routing` | **E3** |
| when the loop stops | `halting` | **E4** |
| depth versus emitted tokens as substitutable compute | latent `loops` vs scratchpad tokens | **E5** |

Two items on the brief's list are deliberately not experiments here.
**Parallel loops / CLP** (§7.1) is deployment engineering: it changes wall-clock
cost, not what depth buys, and its own paper says so. **Interpretability and
safety at depth** (§7.6) is not a separate arm because it is not a separate
model — it is a measurement, so trajectory convergence, state-norm growth,
per-loop exit accuracy and expert-load drift are collected inside E1–E4 rather
than being given a run of their own.

The axes are disjoint (no arm varies two of them) and, together with the two
exclusions above and their stated reasons, they cover the brief's list. That is
what "MECE" has to mean operationally here: each result answers one question,
and no question is left without a result or without an explicit reason.

## 2. Why two task families, and why neither alone would do

Ouro's most valuable finding is a *separation*: storage stays at ~2 bits per
parameter regardless of depth, while composition and multi-hop retrieval scale
with depth and tokens. A study that measures only language-model loss cannot
see that separation — byte-level next-token loss is dominated by exactly the
statistics depth is claimed not to help. A study that measures only a synthetic
composition task cannot claim relevance to language models.

So:

- **`bytes`** — WikiText-2 and PTB, byte level, official splits, bits per byte.
  The storage-and-statistics axis. Prediction: recurrence buys little.
- **`hops`** — in-context composition over a random permutation written into
  the context, scored as accuracy on one answer token at each hop count
  1..4. The manipulation axis, with a knob (`h`) that says how much sequential
  composition an example needs. Fresh graph per example, so nothing can be
  memorised: this measures manipulation with storage held at zero.
- **`twochain`** — the same, but the answer is a function of *two* chains, so a
  partial result must survive while the second chain is computed. This is the
  task E2 needs and it differs from `hops` in one field.

A result that appears on `hops` and not on `bytes` is the separation; a result
that appears on both is something else and would need explaining.

## 3. The arms

Every arm is a `Config`, and every comparison is between configs that differ
in the fields printed in the result file. Where an arm has to differ in two
fields, that is stated in the experiment's header and a control isolates the
second field.

**E0 — baseline and noise floor.** `hops` (2-hop) and `bytes`, R=1, as a **2x2
factorial**: Attention Residuals on/off crossed with MoE/dense. Three seeds on
`hops`, two on `bytes`. Purpose: (i) the seed spread, which every later claim is
read against; (ii) whether the two K3 components survive the scale-down, and
whether they interact. The factorial costs one cell more than two one-at-a-time
ablations and is the only way to see the interaction.

**E1 — depth x data at fixed compute.** `loops` in {1,2,4} x training budget in
{1x, 2x} x hop count in {2, 3}, 2 seeds, on `hops`; a reduced grid on `bytes`.
The brief's question 1 verbatim: *does the useful depth ceiling move with the
token budget?* Read three ways — the argmax over R at each budget (does it
shift right?), the compute-matched diagonal (R x steps constant: is depth or
data the better place to spend a fixed budget?), and the hop axis (P3: whatever
depth buys should buy more of it at 3 hops than at 2, or it is not
composition). Every run is also scored unrolled to depths it was not trained
at, which is Huginn's claim measured for free.

**E2 — where the loop may write.** R=4, `twochain` with `hops` as control,
2 seeds. Four arms:
`plain` (state only) / `+registers` (8 writable positions carried across
loops) / `+registers, wiped each loop` / `+cross-loop AttnRes` (the loop reads
every earlier loop's sublayer outputs). The third arm is the control that
separates *more width* from *width that persists*, which is the actual claim.

**E3 — step-conditioned routing.** R=4, `step_routing` in
{none, bias, embed}, 2 seeds, both task families. Measures accuracy and, more
importantly, whether expert usage actually diverges across loops — a routing
conditioner that changes nothing is worth knowing about. Expert load per loop
is recorded so that the MoE-instability risk the brief flags is observable
rather than assumed.

**E4 — halting.** Trained at up to 4 loops: `none` (fixed R) / `ouro`
(step-indexed gate, uniform prior over T=4) / `pondernet` (step-invariant
gate, geometric prior). Scored at the trained depth *and* unrolled to 8 and 12,
which is the only place the two formulations are predicted to come apart.
Huginn's zero-shot KL exit is measured on the fixed-R model as the
no-training-cost reference.

**E5 — latent depth versus emitted tokens.** Compute-matched: an R-loop model
answering in one token against an R=1 model emitting `h` scratchpad tokens
first. Same task, same parameters, matched forward FLOPs.

## 4. Reading rules, fixed before the runs

1. **Nothing smaller than the seed spread is a result.** E0 measures the
   spread for each task; every later table quotes it, and a difference inside
   it is reported as "inside noise", not as a direction.
2. **Pairing where pairing is available.** Arms share the model-init seed and
   the data stream; on `hops` they also share the evaluation set. Differences
   are read per seed and then aggregated, not aggregated and then differenced.
3. **Compute is reported next to parameters, always.** A looped model is cheap
   in parameters and expensive in FLOPs; every result file carries `params`,
   `active_params`, `flops_per_token` and `train_flops`.
4. **A null on `bytes` is not a null on depth**, and a null on `hops` at these
   sizes is not a null at 1.4B. Both directions of that asymmetry are stated
   with each result.

## 5. What this scale cannot answer, stated up front

- **Parameter efficiency.** Ouro's "1.4B behaves like 4B" is a claim about a
  regime three orders of magnitude away. Nothing here speaks to it.
- **The pipeline.** 7.7T tokens, upcycling, annealing, reasoning SFT — the
  thing that made Ouro competitive — is absent. Depth results here are
  pre-training-only results.
- **Safety monotonicity at depth.** HEx-PHI has no analogue at 0.6M
  parameters. The convergence and norm diagnostics are collected because they
  are the *mechanism* the safety claim would rest on, not as a substitute for
  it.
- **RL on the halting gate** (brief §7.4). It needs sampled trajectories and a
  reward model; on four CPU cores it would be a demonstration, not a
  measurement. E4 gets as far as the supervised gate, and the brief's specific
  claim — that the step-indexed gate with a uniform prior cannot extrapolate —
  is testable without RL and is tested.
- **Anything about 1M context, KDA, or quantisation.** Excluded in
  `BASELINE.md` with reasons.

## 6. Registered predictions

Written before any experiment was run (`git log` for this file is the record).

- **P0 (E0).** Removing either K3 component costs something at this scale. The
  pilot ladder had already found that Full AttnRes is a large *drag* here
  (roughly 1.6x the steps to the same accuracy), which is the opposite sign to
  its own paper's result and is why E0 measures it with seeds rather than
  taking the pilot's word for it.
- **P1 (E1, the central one).** The best R rises with the training budget on
  `hops`. If the argmax over R is the same at 1x and 4x data, the brief's
  central hypothesis is dead at this scale and the answer is that the ceiling
  is a property of the architecture, not of the data.
- **P2 (E1, bytes).** Recurrence buys ~nothing in bits per byte — under half
  the seed spread between R=1 and R=4. This is the separation, and a large gain
  here would mean the loop is buying capacity, not composition.
- **P3 (E1, hop structure).** Accuracy at hop *h* saturates at an R that grows
  with *h*. Any depth effect that is flat in *h* is not composition and would
  have to be explained as something else.
- **P4 (E2).** `+registers` beats `plain` on `twochain` by more than it does on
  `hops`, and `+registers wiped` lands with `plain`. If wiped registers match
  persistent ones, the effect was width, not memory.
- **P5 (E2).** Cross-loop AttnRes is at least as good as registers on
  `twochain`. It is the cheaper mechanism (one vector per sublayer) and it is
  already in the K3 baseline; if it wins, the "nowhere to write" objection is
  partly answered by a component that was there for another reason.
- **P6 (E3).** Step-conditioned routing produces measurable divergence in
  expert usage across loops. Whether it also improves accuracy is a *weaker*
  prediction — the mechanism is expressivity, and at 16 experts and R=4 there
  may be nothing to express.
- **P7 (E3, the honest null).** No routing instability from the loop. The
  aux-loss-free bias is applied once per optimiser step regardless of R, so
  looping multiplies routing decisions per update; if that destabilises load,
  it should show as load variance growing with R.
- **P8 (E4).** The two gates are indistinguishable at the trained depth and
  separate under extrapolation, with the step-invariant/geometric gate
  degrading more gracefully at R=8 and 12. This is the brief's §5 claim and it
  is the only part of it that is cheap to test.
- **P9 (E5).** Scratchpad tokens beat latent depth at matched FLOPs on the
  composition task, because the task's intermediate results are exactly
  representable as tokens. The interesting quantity is the size of the gap, not
  its sign.
