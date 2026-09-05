# Specified and not run

Three things in the brief are specified precisely enough to run and were not
run. Each has a reason, and the reason is not "we ran out of interest".

## E5 — the latent-depth versus emitted-token frontier

**The question.** Depth and tokens are both compute, bought differently: a loop
is sequential and free in context, a scratchpad token is parallelisable and
expensive in context. Nobody has plotted the compute-matched frontier between
them, and both papers leave it open — Ouro's answer is to bolt CoT SFT on
afterwards, which is an admission rather than a measurement.

**The design, in full.** One task, two ways to spend the same FLOPs:

- *latent arm*: `loops=R`, one answer token, loss on that token;
- *token arm*: `loops=1`, the query followed by `h` scratchpad tokens
  `z_1..z_h` (the intermediate entities of the chain) and then the answer, loss
  on all of them, evaluated by generating the `h` tokens autoregressively and
  scoring only the last.

Matched on forward FLOPs per answer: the token arm pays `h` extra positions
through a 4-block model, the latent arm pays `R` core passes over the existing
positions, and `flops_per_token()` in `model.py` already reports both sides.
Hop count `h` is the sweep. Registered prediction (P9 in `DESIGN.md`): tokens
win at matched compute on this task, because its intermediate results are
exactly representable as tokens — the interesting quantity is the size of the
gap, not its sign.

**Why it was not run.** It needs a second task layout (variable-length
scratchpad blocks) and an autoregressive evaluation path, neither of which the
rest of the suite shares, and the CPU budget was already committed to E0–E4.
Adding a half-implemented version would have produced a number nobody should
trust. The task change is about 40 lines in `tasks.py`; the evaluation is a
generate loop in `harness.py`.

## RL on the halting gate (the brief's §7.4)

The gate is the one place in a latent system where credit assignment is
tractable — a discrete action, an explicit cost, a measurable outcome — and the
brief is right that it is unclaimed. It is also the one proposal that cannot be
done honestly here: it needs sampled trajectories per example and enough of
them to estimate an advantage, which on four cores would be a demonstration
rather than a measurement.

What *is* testable without RL is the brief's structural claim — that a
step-indexed gate with a uniform prior cannot extrapolate past its trained
depths — and E4 tests exactly that. The RL step would come after, and only if
E4 shows the supervised gate is worth improving.

## Parallel loops / CLP (the brief's §7.1)

Deliberately excluded rather than deferred. Cross-Loop Parallelism changes the
wall-clock cost of serving a looped model; it does not change what depth buys,
and its own paper says so. Nothing measurable about it exists at this scale:
the latency it attacks is a multi-device scheduling property. The two conflicts
the brief identifies inside its own synthesis architecture — shared-first-loop
KV against writable KV slots, and static CLP scheduling against adaptive
per-token depth — are real, and both are engineering questions that E2 and E4
inform but cannot settle.

## Interpretability and safety at depth (§7.6)

Not parked, but not a separate experiment either: it is a measurement, and it
is collected inside E1–E4 (per-loop exit accuracy, latent trajectory
convergence, expert-load drift with depth). The part that genuinely cannot be
done here is the safety claim — HEx-PHI has no analogue at 0.6M parameters, and
a monotonicity result measured over four steps is not evidence about forty
either way.
