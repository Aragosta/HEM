# RECUR — latent reasoning by depth recurrence, tested on a Kimi K3 block

Working folder for testing the proposals in *Latent Reasoning by Depth
Recurrence* — the comparison of **Huginn** (arXiv:2502.05171, recurrent depth as
a test-time dial) and **Ouro** (arXiv:2510.25741, looping as a pre-training
efficiency argument), and the additions the brief derives from them.

The baseline is a scaled-down **Kimi K3** block: MoE, multi-head attention, and
**Attention Residuals**. Every idea under test is a switch on that one model, so
two arms are two configs rather than two programs.

```
model.py              the K3 block and every recurrence variant, one Config
tasks.py              byte-level WikiText-2/PTB, and in-context composition
spec.py               the scale of the study, in one file
harness.py            training, evaluation, halting rules, the bookkeeping
runner.py             jobs as data; four cores as four single-threaded workers
e0_baseline.py        baseline, 2x2 ablation, and the seed noise floor
e1_depth_data.py      depth x data: does the depth ceiling move with budget?
e2_writable_state.py  does the loop need somewhere to write?
e3_step_routing.py    loop-index-conditioned MoE routing
e4_halting.py         step-indexed vs step-invariant gates, and extrapolation
probe_learnable.py    minimal learnability probe (diagnosed two dead pilots)
probe_reference.py    a textbook transformer, as the "is it my model?" control
BASELINE.md           what is K3's, what was substituted, what is ours
DESIGN.md             the MECE argument, the reading rules, the predictions
RESULTS.md            what the runs found, including the pilots that failed
PARKED.md             E5 and the rest of the brief's list: specified, not run
results/*.json        one file per run, config and metrics included
```

## The question

Both papers iterate a weight-shared block in latent space so that computational
depth becomes a knob independent of parameter count. They disagree about what
that buys. Huginn puts the knob at inference and reports gains as you unroll;
Ouro puts it in pre-training, caps depth at four, and spends its budget on 7.7T
tokens instead. The revealing detail is that Ouro trained at 4–8 steps and
shipped 4 — two independent findings that returns to depth run out early.

The brief's central hypothesis is that the ceiling might be a property of the
*budget* rather than of the architecture, and that the saturation both papers
observed has a candidate explanation neither offered: depth without space. A
loop refines a fixed-width state; it has nowhere to put a partial result. E1
and E2 test exactly those two claims, and E3 and E4 test the two mechanisms the
brief proposes bolting on.

## What is actually measured here

Four CPU cores. Models of 0.2–0.6M parameters, trained for minutes, on a
byte-level corpus and a synthetic composition task. That is enough to measure
*mechanisms* — does depth help composition more than it helps next-byte
prediction, does persistent state beat the same width without persistence, does
a step-conditioned router change routing — and it is nowhere near enough to
speak to Ouro's parameter-efficiency claims or to anything about 1.4B models.
`DESIGN.md` §5 lists what this scale cannot answer, and the results carry that
caveat rather than burying it.

Two task families, because one of them has to be able to stay flat:

- **bytes** — WikiText-2 (and PTB), official splits, bits per byte. The
  storage-and-statistics axis, where Ouro's mechanism predicts depth buys
  little.
- **hops / twochain** — in-context composition over a fresh random permutation
  per example, scored as accuracy on the answer token. The manipulation axis,
  with a hop count that says how much sequential composition an example needs.

## Running it

```bash
pip install torch                              # CPU is fine
python - <<'PY'                                # fetch the corpora
from pathlib import Path
print("see tasks.py: load_bytes() prints the curl command if a split is missing")
PY
python probe_learnable.py --hops 2 --loops 1   # 2 minutes: is anything learning?
python e0_baseline.py --workers 4              # noise floor first, always
python e1_depth_data.py --seeds 0,1 --workers 4
```

Every script takes `--dry-run` (print the jobs and stop) and `--report-only`
(re-render the tables from `results/` without training). Results are skipped if
their file exists, so an interrupted experiment resumes.

## Status

See `RESULTS.md`. The pilots are part of the record: the composition task did
not learn in two earlier forms, and the initialisation of `model.py` was wrong
in a way that cost several hours and is now a comment in the code.

## Sources

- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning*,
  arXiv:2502.05171
- Zhu et al., *Scaling Latent Reasoning via Looped Language Models*,
  arXiv:2510.25741
- Kimi Team, *Attention Residuals*, arXiv:2603.15031, and the Kimi K3 model card
  (https://github.com/MoonshotAI/Kimi-K3)
- Wu et al., *Parallel Loop Transformer*, arXiv:2510.24824 (out of scope here;
  see `DESIGN.md` §1)
- Banino et al., *PonderNet*; Dehghani et al., *Universal Transformers*
