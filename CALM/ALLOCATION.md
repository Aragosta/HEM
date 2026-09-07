# The allocation suite — testing compression-as-objective and its seven consequences

Design document for `T6`–`T12`. Written the way the rest of this repo's suites
are written: the design first, because every result this project has thrown away
was decided before any code ran. Nothing below has been run. Every prediction in
this file is registered *now*, and the runner is required to refuse to interpret
an arm whose gate did not pass.

Companion documents: `suite/README.md` (the 2×2 design and its gates),
`CRITICALITY.md` (the temperature/phase-transition thread this suite inherits),
`SPARSITY.md` (T5, whose null this suite is partly built to explain).

---

## 0. The framing, corrected before it becomes a design

The chain that motivates the suite, in the order the dependencies actually run:

1. **Compression is the objective.** Find the shortest description of the data.
2. **Entropy is the unit.** Bits. Everything measurable here is a bit count.
3. **Energy/compute is the constraint.** ATP for cortex; FLOPs, memory bandwidth
   and parameter storage for a transformer.
4. **Sparsity is the shape the solution takes when that constraint binds** — it
   is a *consequence*, not a mechanism.

Step 4 is the substantive claim and it is the one this suite is built around,
because it has an immediate methodological consequence that has already cost
this project a result: **"add sparsity" is not a hypothesis.** A dense LLM
compresses text extremely well with no sparsity anywhere. T5 spent six seeds ×
three normalisers discovering that every sparsity mechanism it tried was worse
than plain softmax, and the reason is visible in retrospect: none of those arms
had a *binding constraint*. Sparsity imposed where compute is not scarce is pure
capacity loss.

Two independent justifications for step 4 happen to agree, and the suite keeps
them separate because they make different predictions:

* **Solution-side** — constrained optimisation over a polytope puts you at a
  vertex, and vertices are sparse. Predicts sparsity appears *only when the
  budget constraint is active*, and its pattern is whatever the constraint's
  geometry dictates.
* **Data-side** — if the world's latent structure is sparse (few features active
  per scene), the minimal code is sparse regardless of budget. Predicts sparsity
  appears even at a slack budget, and its pattern tracks *the data*.

**T7/T8 test the solution-side story; T10/T11 test the data-side story.** An
experiment that cannot tell them apart is not in this suite.

### The one rule this forces on every arm

Because sparsity is a consequence of a binding constraint, **an arm that is not
budget-matched is not testing anything.** Every comparison below fixes at least
one of {active params/token, FLOPs/token, total params, wall-clock} and reports
all four. T2's MoE arm differed by 0.06% in active params against 22% in total
params — that is the standard, and `budget.py` (§2) enforces it rather than
leaving it to a comment.

---

## 1. What the seven ideas are, and which are already answered here

| # | idea | axis it moves | status in this repo | test |
|---|---|---|---|---|
| 1 | **Conditional width** (MoE / Switch) | parameters ⊥ FLOPs | T2 ran it and got *not resolvable* — 2 seeds, unswept lr, noise ≫ effect | **T7** |
| 2 | **Conditional depth** (MoD / MoR) | compute per *token* | untested | **T8** |
| 3 | **Compression-as-metric** (bits/byte) | none — metrology | `metrics.bits_per_byte` exists, never validated | **T6** |
| 4 | **Route on surprisal**, not a learned gate | routing signal | untested | **T9** |
| 5 | **MDL as the actual objective** | the loss itself | untested | **T10** |
| 6 | **Compressibility-shaped sparsity** (topology) | the attention graph | T5 tested *normalisers*, never *masks* | **T11** |
| 7 | **Residual-only propagation** (predictive coding) | what flows, not how much | untested | **T12** |

Ideas 1–3 are established engineering; the suite's job there is to *measure them
correctly in our hands*, because two of the three have already produced an
uninterpretable number here. Ideas 4–7 are open; the suite's job there is to
build the cheapest experiment that can *kill* each one, and to run the killer
before the builder.

---

## 2. Shared infrastructure — built once, before any arm runs

Six modules. Each exists because a specific past failure in this repo would have
been caught by it. No experiment in §3 may start before its dependencies here
are merged and tested.

| module | what it provides | the failure it prevents |
|---|---|---|
| `suite/budget.py` | `Budget` dataclass: total params, active params/token, FLOPs/token (analytic, not profiled), peak activation bytes, wall-clock/step. `match(reference, arm, on=...)` solves widths to equalise a chosen axis and **raises** if the others drift past a stated tolerance. | T2's MoE-vs-dense read, where a 22% parameter difference sat next to a 0.06% FLOP difference and neither was the axis being discussed. |
| `suite/coding.py` | An actual arithmetic coder. `compress(model, bytes) -> n_bits` and a round-trip decoder. | Reporting bits/byte as "the compressed size" without ever having produced a compressed file. T6 is exactly this check. |
| `suite/routing.py` | `Router` ABC with `signal(x) -> scores`, a hard **capacity** contract (exactly `k` of `n` selected, known before the run), a **causality assertion** for any router that scores across the sequence axis, and instrumentation (selection entropy, load balance, dead-slot count, decision stability across seeds). | MoD-style top-k-over-sequence routing silently leaking future tokens into the selection, which turns a language model into a cheat and shows up as an implausibly good number. |
| `suite/mdl.py` | Description-length accountant: `L_data` (cross-entropy in bits, from `coding.py`), `L_model` (structural bits — active weights under an L0 gate, router table, expert connectivity), and the two-part total. Hard-concrete gates for the differentiable relaxation. | Calling weight decay an MDL prior. It is not one, and T10's whole point is the difference. |
| `suite/graphs.py` | Attention-mask generators (window, window+global, BigBird-random, modular, small-world with rewiring `p`) plus graph metrics: adjacency description length, spectral gap, mean path length, clustering coefficient, degree heterogeneity. | Testing "sparse attention" with one hand-picked pattern and reporting the result as a claim about sparsity rather than about that pattern. |
| `suite/preregister.py` | Writes `{experiment, arms, predictions, falsifiers, seeds, gates}` to a JSON file and hashes it *before* the first training step; the reporter refuses to print a result whose prediction hash is absent or changed. | Reinterpreting a prediction after seeing the numbers. T2's P2 was saved from this only by the author's own note; the machine should enforce it. |

**Instrumentation standard, carried over from T2's methodological finding:**
order parameters reproduced to three significant figures across seeds while
perplexity on the same two runs moved 3.6%. So *every* arm in this suite logs
its internal state — attention entropy, participation fraction, router entropy,
load balance, effective rank, gradient percentiles — at fixed step intervals, and
the analysis reads those first. **More seeds beat more steps** is the allocation
rule; it is not negotiable in this suite either.

---

## 3. The seven experiments

Each block states: the claim, the arms, what is held fixed, the negative control
(the arm that must lose if the effect is real), the metrics, the registered
predictions with their falsifiers, the seed budget, and the gate that decides
whether the next experiment runs at all.

---

### T6 — Is bits-per-byte a metric we are allowed to use?

**Everything downstream is measured in bits, so this runs first and blocks the
rest.** This is a metrology experiment, not an architecture experiment: no
model is being improved, a measuring instrument is being calibrated.

`bits_per_byte` = held-out cross-entropy in base 2, divided by **bytes**, not
tokens. The byte denominator is what makes it comparable across tokenizers; the
claim is that the number is literally the size of the file an arithmetic coder
would emit.

**Arms** — four checks, each of which can independently disqualify the metric:

| check | procedure | pass condition |
|---|---|---|
| **round-trip** | Compress 64 KB of held-out text with `coding.py` driven by the model; decompress; compare bytes. Report emitted bits/byte against the reported metric. | Byte-exact round trip, and emitted bits/byte within **1%** of the analytic figure. A larger gap means the metric is not the compressed size and must be renamed. |
| **tokenizer invariance** | Same model family, three tokenizers: byte, BPE-4000, BPE-16000. | Rank order of the three model sizes is **identical** across tokenizers. Bits/*token* is expected to fail this; that failure is the control that shows the check has teeth. |
| **within- vs across-family correlation** | Spearman ρ between bits/byte and top-1 accuracy + BrierLM, computed (a) within one architecture across sizes, (b) across the five architectures this suite will produce. | Report both. The registered expectation is ρ high within family and **materially lower across**, per recent work; if across-family ρ is also high we have a stronger metric than expected and say so explicitly. |
| **contamination sensitivity** | Deliberately leak 0%, 1%, 5%, 20% of the eval set into training. Measure the bits/byte drop per unit leakage, and the same curve for top-1 accuracy. | A quantified sensitivity curve. This does not pass/fail — it produces the number that later results must be interpreted against, alongside `corpus.overlap_report`. |

**Registered predictions.**

* **P6.1** Round-trip is byte-exact and within 1%. *Falsified by* any decode
  mismatch — which would mean the coder, not the metric, is wrong, and the
  suite stops until it is fixed.
* **P6.2** Bits/byte is tokenizer-invariant in rank order; bits/token is not.
  *Falsified by* bits/token also being invariant, which would mean our tokenizers
  are too similar for the check to discriminate — a null about the *check*, and
  it must be reported as such, not as evidence for the metric.
* **P6.3** Across-architecture ρ is at least 0.2 lower than within-family ρ.
  *Falsified by* the gap being smaller, in which case bits/byte is promoted from
  "training-time signal" to "cross-model comparison metric" for this suite.
* **P6.4 — the honest null.** All four checks can pass and the metric can still
  be a poor guide to downstream capability. T6 licenses bits/byte as a
  *measurement*; it says nothing about whether compression implies capability.
  That question is not answerable at this scale and is not asked.

**Cost.** ~2.0 h (17 short runs for the tokenizer and contamination checks;
the correlation checks reuse T7's models) plus the coder implementation.

**Gate.** If P6.1 fails, **the whole suite halts**: every downstream number is
denominated in a unit we cannot defend. If P6.3 falsifies in the pessimistic
direction (across-family ρ is poor), then bits/byte is used *within* arms only
and every cross-architecture comparison in T7–T12 must carry accuracy and
BrierLM alongside it. That constraint is written into the runner, not left to
the reader.

---

### T7 — Conditional width: does routing buy anything at matched compute, and is the parameter/FLOP decoupling real?

**The claim.** Replacing one FFN with `N` experts and a top-`k` router makes
parameter count and FLOPs/token independent axes. A model with 50× the
parameters at the same compute per token stores many specialised codebooks and
selects one — storage is cheap, decode is expensive, so you buy storage.

**Why re-run this.** T2 tested it and got *not resolvable*: 2 seeds, an unswept
learning rate, 700 steps, and per-seed paired differences of −5.18 / +8.39 with
the sign flipping. That is a design failure, not a result. The corrected design
fixes all three known causes and adds the axis T2 never varied.

**Arms — a 2×2×3, because the interesting claim is about two budget axes at once.**

|  | matched **FLOPs**/token | matched **total params** |
|---|---|---|
| **dense** | reference | reference (same model; it is the corner where the two matchings coincide) |
| **MoE** | more params, same compute — *the claim* | same params, less compute — *the mirror claim* |

crossed with **expert count `N` ∈ {4, 16, 64}** at fixed `k=2`+1 shared, which is
the axis that actually tests decoupling: as `N` grows at fixed `k`, the
parameter/FLOP ratio grows linearly while compute per token is constant.

**Held fixed.** Data order, seed, optimizer, schedule, `lr` (swept once, in
advance, on the dense arm — the T2 defect), sequence length, depth, `dim`.
Expert width is set by `expert_width_for` so the FLOP-matched cells are matched
by construction, and `budget.py` asserts it rather than trusting the arithmetic.

**Negative control — the one that makes the test exclusive.** A **random-router**
arm: identical architecture and identical budget, router logits replaced by a
fixed random projection that is *not trained*. If MoE's benefit is real
specialisation, MoE must beat random routing. If it does not, the effect is
capacity-from-many-experts (or ensembling), not conditional computation, and the
paper-standard framing is wrong in our hands. **T2 had no such control, so its
null could not have been diagnosed even if it had been resolvable.**

**Metrics.** bits/byte (T6-licensed), top-1, BrierLM; `router_entropy_norm`,
`load_balance`, `dead_experts` (already instrumented); expert-specialisation
index (mutual information between token identity and expert choice); and the
**structured-sparsity claim** measured directly — wall-clock per token as `N`
grows, which is the whole practical argument for MoE over unstructured sparsity.

**Registered predictions.**

* **P7.1** At matched FLOPs, MoE beats dense in bits/byte, and the margin grows
  monotonically with `N`. *Falsified by* a flat or non-monotone `N` trend, which
  would say the decoupling is not being converted into quality at this scale.
* **P7.2** MoE beats the random-router control at every `N`. *Falsified by*
  parity — which would be the most informative outcome in this experiment and
  must be reported as the headline, not a footnote.
* **P7.3** At matched *total params*, MoE loses to dense (it is doing less work
  per token). *Falsified by* MoE winning both matchings, which would mean the
  gain is not about budget at all and something in the matching is wrong.
* **P7.4** Wall-clock/token is flat in `N` up to `N=64` at fixed `k`. This is the
  structuredness claim; *falsified by* super-linear slowdown, which for our naive
  loop implementation is a real possibility and would make the "free capacity"
  claim an implementation artefact of production kernels, not a property of the
  method. Reported either way.

**Seeds.** 6 per cell, paired by construction (shared init for everything except
the router and expert tensors, which cannot be shared; pairing is therefore on
*data order and non-FFN weights*, and `assert_paired` is relaxed accordingly and
documented as weaker pairing than T3/T5 enjoyed). Power: T5's paired-difference
sd of 3.38 at 6 seeds resolves a ~4% effect at t=5.65; a 1% effect needs the
tighter pairing T7 cannot have, so **T7 is powered for ≥3% and says so up front.**

**Cost.** 13 cells × 6 seeds = 78 runs, **~22 h — the largest item in the
suite, and 15 h of it is the `N=64` arm alone.** That is an artefact of
`MoE.forward` being a Python double loop over `(slot, expert)`: measured cost is
0.155 s/step dense against 0.215 s at `N=4`, i.e. ~6.7 ms per mask-expert op,
which at `N=64` means 129 of them per layer per step. **Vectorising the expert
loop before T7 runs cuts this to ~8 h** and is also the precondition for P7.4
being a test of the method rather than of our implementation. Tier 2 escalation
only if P7.2 holds.

**Gate.** If P7.2 fails (routing ≈ random routing), T8 and T9 are both
*conditional-computation* experiments resting on the premise that a learned
router does something, and both must be re-scoped to ask whether *any* routing
signal beats random before asking which signal is best.

---

### T8 — Conditional depth: fixed budget, fluid allocation

**The claim.** Mixture-of-Depths applies the same trick to the depth axis: at
each layer a router scores every token and only the top-`k` get the full
attention+FFN; the rest skip via the residual. Because `k` is fixed in advance,
total FLOPs are known before the run, but *which* tokens consume them is decided
at runtime from context. Mixture-of-Recursions changes the axis rather than the
mechanism — one weight-tied block, a router deciding how many times to apply it.

**Why the fixed budget is the design, not a detail.** Early-exit schemes let each
token halt when it wants, so compute per batch is unpredictable and the hardware
idles on stragglers. MoD keeps the budget rigid and lets only the assignment
move. That is the transportation-polytope structure from §0: **the vertex is
sparse because the constraint binds.** T8 is therefore the cleanest available
test of the solution-side story.

**Arms.**

| arm | mechanism | budget |
|---|---|---|
| `dense` | every token through every layer | reference |
| `mod` | per-layer top-`k` token routing, capacity ratio `c ∈ {0.125, 0.25, 0.5, 0.75, 1.0}` | FLOPs set by `c`; at `c=1.0` it must be *bit-identical* to dense — a built-in correctness assertion |
| `mod-random` | **negative control**: same capacity, tokens chosen by a fixed random mask | identical |
| `mod-frequency` | **second control**: tokens chosen by a static heuristic (token unigram rarity), no learning, no context | identical |
| `early-exit` | per-token halting, no capacity constraint | *variable* — and its variance in FLOPs/batch is the number being measured |
| `mor` | weight-tied block, router picks recursion depth 1–3 per token | matched to `mod` at equal mean depth |

**The causality audit, run before any number is believed.** Top-`k` over the
sequence axis is a comparison *between tokens*, so a naive implementation lets
token `t`'s selection depend on tokens `> t`. `routing.py` asserts causality by
construction and T8 additionally runs the empirical check: train with
teacher-forced routing, evaluate with a causal predictor of the routing decision,
and report **both** numbers. A large gap is the leak, and the leaky number is
never reported alone.

**Metrics.** bits/byte at matched FLOPs; FLOPs/token *variance* (the argument
against early-exit); wall-clock; and the allocation itself — **which tokens get
compute**, cross-tabulated against token surprisal, position, part-of-speech
proxy, and whether the token is inside a repeated n-gram. That cross-tab is the
bridge to T9 and is worth more than the perplexity column.

**Registered predictions.**

* **P8.1** `mod` at `c=0.5` matches dense bits/byte within noise while using
  ~50% of the layer FLOPs. *Falsified by* a gap exceeding the T7-established
  noise floor.
* **P8.2** `mod` beats `mod-random` and `mod-frequency` at every `c < 1`. This is
  the exclusivity condition: without it, "adaptive depth" is indistinguishable
  from structured dropout at reduced compute.
* **P8.3** `early-exit` matches `mod` in quality but shows ≥3× the FLOPs/batch
  variance. *Falsified by* low variance, which would remove the main published
  argument for the capacity constraint.
* **P8.4** The compute allocation correlates positively with token surprisal
  measured by a *separate* frozen reference model. This is the prediction that
  makes T9 worth running; *falsified by* ρ ≈ 0, which would say learned routers
  are doing something other than "spend where prediction is hard" and would make
  T9's whole premise doubtful before it costs anything.
* **P8.5** `mor` reaches `mod`'s quality at fewer parameters (weight tying) but
  needs more steps. Reported as a params/steps tradeoff curve, not a winner.

**Seeds.** 5 per cell; `c` sweep at 3 seeds for the interior points, 5 at
`c ∈ {0.25, 0.5}` where the decision lives.

**Cost.** ~5.4 h (72 runs; MoD is cheaper than dense per step at `c < 1`, and
`mor` costs ~2× for the recursion). `mor` adds a tied-weight model to `hybrid.py`.

**Gate.** P8.4 gates T9. If compute allocation does not track surprisal at all,
T9 is re-scoped from "is surprisal a better signal" to "what *is* the learned
router's signal", which is a probing experiment and much cheaper.

---

### T9 — The routing signal: surprisal vs a learned gate, and the epistemic/aleatoric confound

**The claim.** Today's router is a small learned linear map whose meaning is
unknown — it is whatever minimised loss. Predictive coding says the principled
signal is already present: allocate compute where prediction is failing.
Concretely: the entropy of the output distribution at that position, the norm of
the residual update, or the KL between successive layers' implied predictions.
It would make the router non-parametric and interpretable, and it is the rule
cortex appears to use.

**The objection this experiment is built around, and which makes it interesting
rather than routine.** Surprisal conflates two things:

* **epistemic** — the model lacks the reasoning to pin the token down. More
  compute helps.
* **aleatoric** — the token is genuinely unpredictable (an arbitrary proper
  noun). More compute is *wasted*.

Naive surprisal routing spends the budget on irreducible noise. What you want is
**reducible** uncertainty, and estimating that requires knowing how much the
prediction would improve with more compute — which is the thing you were trying
to avoid computing. There is also a chicken-and-egg problem: the cleanest error
signal comes *after* the layer runs; the routing decision comes *before*.

**Arms.** All at identical capacity `c=0.5`, identical budget, paired seeds.

| arm | signal | costs |
|---|---|---|
| `learned` | trained linear gate (the T8 `mod` arm) | reference |
| `entropy` | output-distribution entropy at the previous layer's implied prediction | ~0 params |
| `residual-delta` | ‖Δx‖ of the previous layer's residual update — the causally-available proxy for "this token is still moving" | 0 params |
| `layer-kl` | KL between consecutive layers' implied next-token distributions | one shared readout |
| `oracle-total` | **upper bound**: offline, route by the *actual* surprisal under a fully-trained reference model | not deployable — that is the point |
| `oracle-reducible` | **the real upper bound**: offline, route by measured *improvement* from extra compute, computed by running each token both ways | not deployable |
| `random` | fixed random mask at the same capacity | the floor |

**Why the two oracles are the heart of the design.** They are not competitors;
they bracket the achievable. `oracle-reducible` is the ceiling any surprisal-like
signal could reach. `oracle-total` is what a *perfect* naive-surprisal router
gets. **The gap between them is the exact cost of the epistemic/aleatoric
confound, measured rather than argued** — and if that gap is small, the objection
that motivates this whole experiment is quantitatively unimportant and the cheap
non-parametric routers become attractive. If it is large, no deployable surprisal
signal can be good, and the finding kills the idea cleanly at a cost of one
offline run.

**Metrics.** bits/byte; the two gaps (`learned − random`, `oracle-reducible −
oracle-total`); routing agreement matrix (Jaccard of selected-token sets between
every pair of arms — this says whether the learned gate has *discovered*
surprisal); and a decomposition of allocated compute against a
heteroscedasticity proxy for aleatoric content (token-type entropy under the
bigram table already built in `metrics.lookup_baselines`).

**Registered predictions.**

* **P9.1** `oracle-reducible` > `oracle-total`, and the gap exceeds the
  `learned − random` gap. Meaning: the confound costs more than the router buys.
  *Falsified by* the gaps being comparable, which would license deployable
  surprisal routing as a real candidate.
* **P9.2** Among deployable signals, `residual-delta` ≥ `entropy`, because it is
  causally available at decision time and does not require the layer's own
  output. *Falsified by* `entropy` winning, which would mean the timing problem
  is less binding than the signal-quality problem.
* **P9.3** No deployable non-parametric signal beats `learned`. This is the
  registered *negative* prediction and the honest prior; the experiment is run
  because the *gap structure*, not the winner, is what is informative.
* **P9.4** The learned gate's selections agree with `oracle-total` above chance
  but well below `oracle-reducible`. That would be the mechanistic statement:
  learned routers approximate naive surprisal, including its mistake.

**Seeds.** 5 per arm; the two oracle arms need 3 (they have no router variance,
only seed variance in the underlying model).

**Cost.** ~3.1 h (31 runs), of which ~1 h is the offline oracle pass — every
token evaluated both ways, but at eval cost, not training cost. **The cheapest
experiment in the suite relative to what it can kill.**

**Gate.** P9.1 decides whether idea 4 is dead. Either outcome is publishable
internally; only "we didn't measure the confound" is not.

---

### T10 — MDL as the actual objective

**The claim.** The two-part code says total description length =
`L(model) + L(data | model)`. Training optimises only the second term; weight
decay is a crude and unprincipled stand-in for the first. A real MDL objective
charges the model for its own structure: how many weights are active, how the
expert connectivity is wired, how many bits the routing table costs.

**Why this is the interesting one.** It collapses three separate questions into
one optimisation. Pruning becomes automatic — an edge survives iff its
contribution to `L(data|model)` exceeds its own `L(model)` cost. Routing sparsity
becomes automatic for the same reason. And criticality comes free, because
MDL-optimal codes sit at a second-order phase transition by construction: you
would not *tune* to criticality, you would land there.

**Why it is hard, and why our scale is the right scale to try it.** At LLM scale
the arithmetic is unkind: `L(data|model)` is in terabits and `L(model)` in
gigabits, so the structural term barely moves the gradient unless you weight it
artificially — at which point you have abandoned the principle and are back to
tuning a hyperparameter. **At tier-1 scale the ratio is far more favourable**
(≈5M params against ≈900KB of bytes), so a toy is informative here in exactly the
way the large model is not. That is an unusual and genuine advantage of this
repo's scale and the suite should exploit it deliberately.

**Arms.**

| arm | `L(model)` term | `λ` |
|---|---|---|
| `none` | — | 0 (reference) |
| `wd` | weight decay 0.01 | the current default; the thing MDL claims to replace |
| `l0-principled` | expected active-weight count via hard-concrete gates, priced at the true bits/weight | **λ = 1 exactly** — the MDL principle admits no free constant |
| `l0-tuned` | same | λ swept ∈ {0.1, 0.3, 1, 3, 10} |
| `l0-structural` | active weights **+ router table bits + expert connectivity bits** | λ = 1 |
| `l0-shuffled` | **negative control**: gates present and trained, but each gate's cost is a random permutation of the true per-weight costs | λ = 1 |

**The `λ = 1` arm carries the experiment.** If MDL is a principle rather than
another regulariser, λ = 1 must be at or near the optimum of the λ sweep. If
the swept optimum is far from 1, MDL as implemented is a regulariser with a
suggestive derivation, and that is the finding.

**Metrics.** Total description length in bits (the objective itself, reported as
the headline — this is the only experiment in the suite whose objective and
metric coincide, which is the point); bits/byte on held-out data; achieved
sparsity fraction and *where* it lands (per-layer, attention vs FFN, router vs
expert); and the **criticality check** — `entropy_norm` and `participation_frac`
from `t2_criticality.measure`, to test whether the MDL-trained model sits at the
operating point T2 found all four architectures converging to (`participation_frac`
0.20 ± 0.02, `entropy_norm` 0.41 ± 0.03).

**Registered predictions.**

* **P10.1** `l0-principled` (λ=1) beats `wd` on total description length. This is
  nearly definitional — `wd` is not optimising that objective — so it is a
  *sanity* prediction, and its failure means the accountant is wrong.
* **P10.2** `l0-principled` beats `wd` on **held-out bits/byte**. Not
  definitional at all, and the real test. *Falsified by* `wd` winning, which
  would say the structural prior is a worse inductive bias than an arbitrary one.
* **P10.3** The λ sweep's optimum lies within a factor of 3 of λ=1. *Falsified by*
  an optimum at 10 or 0.1, which demotes MDL from principle to heuristic here.
* **P10.4** MDL-trained models land at or nearer the T2 attractor
  (`participation_frac` ≈ 0.20) than the `none` arm, without being tuned to.
  This is the criticality-comes-free claim, and it is the one I would most expect
  to fail, because T2's finding was that *every* architecture lands there anyway
  — in which case the correct report is "the attractor is not diagnostic",
  not "MDL found criticality".
* **P10.5** `l0-structural` prunes the router table before it prunes expert
  weights, because the router is cheap to describe and expensive to justify.
  Directional; a sharp mechanistic prediction that costs nothing extra to check.

**Seeds.** 6 for the three headline arms (paired: gates are additive parameters
and the shared weights can be initialised identically, so this recovers the
strong T3/T5-grade pairing and therefore ~1% resolution). 3 per λ point.

**Cost.** ~6.5 h (45 runs plus the two §4.2 cross cells). `mdl.py` is the
largest new module, and the L0 gates cost ~15% per step.

**Gate.** P10.2 is the decision. If the principled objective loses on held-out
bits/byte, ideas 5 and 6 collapse into each other — because T11's topology claim
is downstream of "structure has a price" — and T11 should be run as a pure
empirical mask sweep with no MDL framing.

---

### T11 — Attention topology: the compressibility/mixing tradeoff

**The claim.** Network compressibility results give a target topology: networks
are compressible when they have high transitivity and heterogeneous degrees —
modular clusters plus heavy-tailed hubs. Current sparse attention does not derive
its pattern from anything; sliding-window-plus-a-few-global-tokens is a
reasonable guess dressed as a design. The proposal is to shape the attention
graph to have the hierarchical-modular, heavy-tailed structure the theory says
is optimal.

**The tension that makes it an experiment rather than an implementation.**
Compressibility and mixing are in conflict. BigBird-style patterns include
*random* edges precisely because random graphs are good expanders — short paths,
fast mixing, provable approximation of full attention. But random graphs are also
maximally *incompressible*; that is what randomness means. So the compressible
topology (modular, clustered) has poor mixing, and the well-mixing topology has
no compressible structure. Small-world architecture — mostly clustered, a few
long-range shortcuts — is the brain's answer, and where the optimum sits for
attention specifically is unknown.

**Arms — a rewiring sweep, because the tradeoff is a one-dimensional family.**
Watts–Strogatz over the causal attention mask: start from a pure local window
(`p=0`, maximally clustered, maximally compressible, worst mixing) and rewire a
fraction `p` of edges to random long-range targets (`p=1`, an expander, best
mixing, incompressible).

`p ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0}` at **fixed edge count**, so density is not
the variable — only topology is. Plus three named comparators at the same edge
budget: `window+global` (the current standard), `bigbird` (window + global +
random), `modular` (hierarchical block structure, no shortcuts), and `dense`
(full attention, the unconstrained ceiling; not budget-matched, and reported as
a ceiling rather than an arm).

**The task battery, because one task cannot see this tradeoff.** Mixing only
matters if the task needs it:

| task | needs | source |
|---|---|---|
| local LM | short-range only | `corpus.load` as usual |
| needle retrieval | one long-range hop | synthetic, `data_hierarchy.py` style |
| multi-hop retrieval | ≥2 long-range hops — **this is where mixing must show up** | synthetic |
| hierarchical parity/agreement | modular structure | `data_hierarchy.py` |

**Metrics.** bits/byte or task accuracy per task; and, for every mask, the graph
quantities from `graphs.py` — adjacency description length (bits), spectral gap,
mean path length, clustering coefficient, degree heterogeneity. **The headline
plot is task performance against adjacency description length, one point per
mask**, which is the compressibility/performance frontier stated directly.

**Registered predictions.**

* **P11.1** The optimum over `p` is **interior** — better than both `p=0` and
  `p=1` — on the multi-hop task. This is the small-world claim and the reason to
  run the sweep rather than three named masks. *Falsified by* a monotone curve,
  which would mean one of the two ends simply wins and the tradeoff is not
  balanced at this scale.
* **P11.2** `p*` is small (≤0.1): a few shortcuts buy most of the mixing. Direct
  from the small-world literature and cheap to check.
* **P11.3** On the local-LM task the curve is flat in `p` — no mixing is needed,
  so no topology is better. *Falsified by* a strong `p` dependence, which would
  mean topology is doing something other than governing information flow and the
  whole framing is wrong.
* **P11.4** `bigbird` ≈ small-world at matched `p`; the random edges *are* the
  shortcuts, and the two designs are the same object under different names.
* **P11.5 — the honest limit, carried directly from T5's P4.** At `seq_len 128`
  there is very little long-range structure for any topology to exploit. A null
  here is weak evidence and must be reported as "not at this context length",
  never as "topology does not matter". If P11.1 is null, **the informative
  follow-up is the length axis, not more seeds** — and the length sweep is
  specified in advance (§5) so that decision is not made post hoc.

**Seeds.** 4 per mask per task (14 masks × 4 tasks is already large; the seed
budget goes to the multi-hop task where the prediction lives — 6 there, 3
elsewhere).

**Cost.** ~12.9 h (10 masks × 4 tasks, seeds concentrated on multi-hop where the
prediction lives) — the second-largest item, and the one that grows fastest if
the P11.5 length follow-up is taken. Masks are static buffers, so no kernel work is required at this scale; the *efficiency* claim of
sparse attention is explicitly **not** tested here (dense masking of a sparse
pattern costs the same as dense attention) and that limitation is stated in the
module docstring, not discovered by a reader.

---

### T12 — Residual-only propagation

**The most radical of the seven, and the only one that changes *what* flows
rather than *how much*.**

**The claim.** A transformer's residual stream carries the full representation;
each block reads it, computes, and adds its output back. Cortex under predictive
coding does something different: superficial pyramidal cells send prediction
*error* forward, deep pyramidal cells send predictions backward, and the forward
signal is suppressed to the extent the top-down prediction was correct. What
propagates up is the residual in the compression sense — the part that could not
be explained. A transformer built this way would have layer `L+1` receive only
what layer `L` failed to predict about it. That is a strictly better code *if*
the layers are predictable from each other, and adjacent-layer representations
are known to be highly similar in deep models.

**Why it is close to disqualifying.** Predictive coding requires iterative
settling: error and prediction are mutually defined, so you relax to a fixed
point over several passes per input. That destroys the single-forward-pass
parallelism that makes transformer training economical. It also needs a top-down
pathway, roughly doubling parameters. Learning is *not* the obstacle — predictive
coding networks can approximate backprop — throughput is. This is the same wall
neuromorphic approaches keep hitting.

**So T12 is a three-stage ladder with kill gates, and stage A costs almost
nothing.**

**Stage A — measurement, no architecture change.** Take the trained models T7
already produced. For each adjacent layer pair, fit a linear (and a low-rank)
probe predicting layer `L+1`'s input from layer `L`'s, and report the explained
variance `R²` and the *residual entropy* — how many bits are actually new at
each layer. This is the premise of the whole idea, stated as a number, and it
costs one afternoon.

* **Gate A.** If mean `R²` < 0.5, adjacent layers are not redundant, there is no
  code to save, and **T12 stops here** with a clean negative that also explains
  why: the transformer's residual stream is not the wasteful object the analogy
  assumes.

**Stage B — one-pass approximation, no settling.** Predict-and-subtract: a cheap
top-down predictor `g_L` estimates layer `L+1`'s input from layer `L`; the forward
signal carries `x_{L+1} − g_L(x_L)`; the prediction is added back where needed.
Single forward pass, so the parallelism survives; this is the deployable version
and it captures the *coding* claim without the *dynamics* claim.

* Arms: `dense` (reference), `stage-b` at predictor ranks {1, 4, 16, full},
  `stage-b-frozen` (predictor at init, never trained — **negative control**:
  isolates whether the benefit is the learned prediction or merely the subtraction
  and its effect on the residual stream's scale), and `stage-b-shuffled`
  (predictor from a different layer — controls for "any subtraction helps").
* **Gate B.** If `stage-b` does not match `dense` at matched budget, stage C is
  not run: the code is not better even without the throughput penalty, and adding
  iterative settling cannot fix a coding loss.

**Stage C — full predictive coding, tiny scale, throughput measured as the
headline.** Iterative settling with `T ∈ {2, 4, 8}` relaxation steps and a full
top-down pathway. Quality is a secondary metric here; the primary output is the
**quality-per-wall-clock frontier against the dense baseline**, because the
honest question is not "does it work" but "does it ever pay for `T` passes".

**Registered predictions.**

* **P12.1** Adjacent-layer `R²` is high (>0.7) in the middle layers and lower at
  the first and last. Consistent with the known similarity result and cheap.
* **P12.2** `stage-b` matches dense within noise but does **not** beat it. The
  redundancy is real but the residual stream is already carrying it cheaply;
  removing it saves description length that was not costing anything.
* **P12.3** `stage-b-frozen` ≈ `stage-b`. If this holds, the benefit (if any) is
  from rescaling the residual stream, not from prediction — a deflationary result
  and the single most likely way this idea turns out to be nothing.
* **P12.4** `stage-c` needs `T ≥ 4` to match dense, at ≥4× the wall-clock, so it
  loses the frontier decisively. *Falsified by* `T=2` sufficing, which would make
  the throughput objection a factor of two rather than a wall, and would be the
  one result in this suite that justifies real GPU spend on idea 7.

**Seeds.** Stage A: no training. Stage B: 5 per arm, paired (the predictor is
additive, so shared init is exact — strong pairing, ~1% resolution). Stage C: 3.

**Cost.** Stage A ~1.5 h (no training — probes on T7's models). Stage B ~3.5 h.
Stage C ~3.6 h, and only if B passes. **The gate structure means the expected
cost is 1.5 h, not 8.6 h**, because Gate A is the likeliest place this stops.

---

## 4. Cross-cutting design

### 4.1 Run order and dependencies

```
T6  (metrology gate) ──┬──> T7 (conditional width) ──┬──> T8 (conditional depth) ──> T9 (routing signal)
                       │                             │
                       ├──> T10 (MDL objective) ─────┴──> T11 (topology)
                       │
                       └──> T12-A (redundancy probe, uses T7's models) ──> T12-B ──> T12-C
```

T6 blocks everything. T7's random-router control gates T8/T9's premise. T8's
P8.4 gates T9's scope. T10's P10.2 gates T11's framing. T12-A gates T12-B gates
T12-C. **No experiment in this suite may be started because its predecessor is
"probably fine".**

### 4.2 Interactions that are worth a cell, and one that is not

The suite is factorial only where a mechanism connects the factors. From T5's
P3 — which held — attention sparsity and MoE routing reduce *different graphs*
and do not interact. The same reasoning applies here, and it is applied
explicitly so that cells are bought for a reason:

| pair | mechanism connecting them? | cell bought? |
|---|---|---|
| T7 × T8 (width × depth routing) | yes — both spend the same FLOP budget, and a token skipped at depth cannot use its expert | **yes**, a 2×2 at `c=0.5, N=16` |
| T9 × T8 (signal × depth) | by construction — T9 *is* T8 with the signal swapped | merged, not crossed |
| T10 × T7 (MDL × routing) | yes, and it is the strongest claim in the suite: MDL should **derive** the routing sparsity T7 imposes by hand | **yes** — `l0-structural` on an MoE model, and compare its discovered `k` against T7's imposed `k` |
| T10 × T11 (MDL × topology) | yes — MDL should derive the mask | **yes**, one arm: learn the mask under `l0-structural` and measure its adjacency description length against T11's frontier |
| T11 × T7 (topology × experts) | no mechanism; token-token graph vs token-expert graph, exactly T5's P3 | **no** |
| T12 × anything | stage B changes the residual stream, which every other mechanism reads | deferred; if stage B passes, the crosses are re-scoped as new work |

The two MDL crosses are, in my judgement, the highest-value cells in the whole
suite: they are the only places where the framing in §0 makes a *falsifiable*
prediction about a mechanism it did not design.

### 4.3 Statistics

Carried directly from what this repo has already measured, not from convention.

* **Pairing is mandatory where it is possible.** T3 showed pairing made a 1%
  effect measurable that the unpaired design would have needed ~79 seeds to see.
  Each experiment above states its pairing strength honestly: strong (additive
  parameters, bit-identical shared init — T10, T12-B), weak (shared data order
  and non-varying weights only — T7, T8), or none (T11 masks change the
  computation graph).
* **Resolution by pairing class**, from the measured paired-difference sd of
  3.38 in T5: strong pairing at 6 seeds resolves ~1%; weak pairing at 6 seeds
  resolves ~3–4%. **Every experiment states the effect size it is powered for
  before it runs**, and an effect below that threshold is reported as "not
  resolvable at this power", never as a null.
* **Tests.** Paired t against the `T_CRIT` table already in `t5_sparsity.py`,
  plus a sign test, and both reported. A result where the two disagree is
  reported as unresolved.
* **The seeds-over-steps rule.** T2: order parameters reproduced to three
  significant figures while perplexity moved 3.6% on the same runs. Marginal
  compute goes to seeds until the paired sd is measured, then to steps.
* **Non-finite steps invalidate a row outright.** Three results in this project
  were numerical failures wearing the costume of an architectural finding.
* **Floor gate.** Any arm scoring below the bigram lookup table has not learned
  the corpus and its row is descriptive of nothing. Non-negotiable after a
  previous run scored 9.23% held-out against a bigram table's 19.86%.

### 4.4 Tiers

| tier | hardware | scale | what it decides |
|---|---|---|---|
| **0 — smoke** | CPU, minutes | ~30 steps | every arm runs, every metric finite and non-vacuous, every gate reachable |
| **1 — CPU study** | 4 CPU cores, hours | ~5M params, BPE-16000, seq 128 | signs, noise floors, and every gate in §4.1. **This is where the suite lives.** Not a claim about language models at scale |
| **2 — single GPU** | 1 GPU, hours | 120M preset, seq 1024 | only the arms that passed their tier-1 gate, and only the ones whose claim is *length-dependent* (T11 above all) |
| **3 — comparable** | multi-GPU | published settings | not planned; listed so that the absence is deliberate |

### 4.4.1 The cost model, and where the budget actually goes

Not an estimate — measured. `t1/t2/t5_results.json` record `ms_per_step` for the
exact configuration this suite uses (dim 192, 4 layers, head_dim 32, kv_latent
48, seq 128, batch 8, BPE-16000, 4 CPU cores):

| configuration | measured | per run @ 2000 steps |
|---|---|---|
| dense / softmax | 0.155 s/step | 5.2 min |
| MoE `N=4`, `k=2`+1 shared | 0.215 s/step | 7.2 min |
| MoE `N=16` (extrapolated, same loop) | 0.375 s/step | 12.5 min |
| MoE `N=64` (extrapolated, same loop) | 1.01 s/step | 33.8 min |
| entmax (not used in this suite) | 0.394 s/step | — |

**2000 steps is the suite standard**, chosen deliberately: T2 ran 700 steps and
called it out itself as 0.30 epochs, under-trained, and a reason its perplexity
column could not be read. 2000 steps is ~0.86 epoch on this corpus. Every figure
below scales linearly in that choice.

| item | runs | hours | share |
|---|---|---|---|
| tier-0 smoke, all arms | — | 1.0 | 1.6% |
| lr sweep (5 lr × 3 seeds, in advance — the T2 defect) | 15 | 1.3 | 2.1% |
| **T6** metrology | 17 | 2.0 | 3.1% |
| **T7** conditional width | 78 | **21.9** | **35.0%** |
| **T8** conditional depth | 72 | 5.4 | 8.6% |
| **T9** routing signal | 31 | 3.1 | 5.0% |
| **T10** MDL objective | 57 | 6.5 | 10.4% |
| **T11** topology | 150 | **12.9** | **20.6%** |
| **T12** residual coding (A+B+C) | 44 | 8.6 | 13.7% |
| **total** | **464** | **62.7** | |

Total tier-1 budget: **~63 CPU-hours nominal, ~81 h with a 30% contingency for
re-runs and non-finite rows.** Three things move that number materially:

* **Vectorise `MoE.forward` first.** The Python `(slot, expert)` loop is 35% of
  the suite's entire budget and all of T7's `N=64` cell. A batched
  `index_add`/einsum implementation takes the suite to **~48 h nominal, ~63 h
  with contingency** — a saving larger than any other decision available here,
  and it is a day of work.
* **Steps.** At the old 700-step setting the suite is ~22 h, but T2 already
  established that 700 steps cannot resolve a perplexity comparison at this
  scale. Buying the shorter run buys an unreadable result; this is the one place
  the budget should not be cut.
* **Gates.** The §4.1 DAG is designed so failures are cheap. If Gate A of T12
  fails, 7 h evaporates; if T6's round-trip fails, the whole 63 h stops on day
  one. **Expected** cost is meaningfully below nominal, and that is deliberate.

**"CPU-hours" here means wall-clock hours of one run at a time**, since torch is
already threading across the 4 cores — the measured `ms_per_step` figures include
that. Running the DAG's four independent branches concurrently requires
`OMP_NUM_THREADS=1` per process, which raises per-run time roughly 2–3× and so
yields a net throughput gain of only ~1.5×, not 4×. Realistically: **~8 days of
wall-clock serial, ~5 days with the branches parallelised, ~4 days if the MoE
loop is vectorised first.**

### 4.5 The traps this suite is explicitly built to avoid

Each is a failure that has already happened in this repo:

1. **Comparing arms that are not budget-matched** (T2's params-vs-FLOPs) →
   `budget.py` raises rather than reports.
2. **No negative control**, so a null cannot be diagnosed (T2's MoE arm) → every
   experiment above has a named control arm that must lose.
3. **Reinterpreting a prediction after the numbers** → `preregister.py` hashes
   predictions before step 0.
4. **A metric reported in a cell where it does not exist**, or two metrics
   compared that are not on the same scale → the metric-to-cell table is part of
   each module's docstring, as in the 2×2 suite.
5. **Weak evidence reported as a null** (T5's P4, which was flagged correctly and
   is copied here as T11's P11.5) → each experiment names in advance the axis its
   null would point to.
6. **Under-powered comparisons quoted as results** (T2's entire perplexity
   column) → powered-for effect size stated per experiment.
7. **Instrumentation added after the run** → order parameters logged from step 0
   in every arm, because they have already proven more sensitive than the loss.

### 4.6 What this suite cannot answer

Stated here so it cannot be quietly forgotten in a results table:

* **Nothing at scale.** Tier 1 is ~5M parameters on ~900KB of text. Architecture
  comparisons at this scale are suggestive of sign, not of magnitude, and are
  known to invert for some mechanisms.
* **Nothing about long context.** `seq_len 128`. This directly weakens T11
  (topology) and T8 (which tokens need compute), and both say so in their
  registered predictions rather than in a caveat at the end.
* **Nothing about efficiency in wall-clock terms for sparse attention.** T11's
  masks are dense buffers; the compute saving of a sparse pattern requires
  kernels this suite does not write.
* **Nothing about whether compression implies capability.** T6 licenses
  bits/byte as a *measurement*. The inference from "compresses better" to "more
  capable" is a separate claim, degrades across architectures, and is out of
  scope.

---

## 5. Pre-registered follow-ups, so that post-hoc choices are not made post hoc

If a null arrives, the next experiment is already decided:

| null | follow-up, registered now |
|---|---|
| T7 P7.2 (MoE ≈ random routing) | Increase `N` to 256 before increasing seeds — the decoupling claim is about the params/FLOPs ratio, and 4 experts may simply be too few for specialisation to exist |
| T8 P8.2 (`mod` ≈ `mod-random`) | Length axis, not seeds: adaptive depth should matter more when tokens differ more, and at seq 128 they do not |
| T9 P9.1 (gaps comparable) | Build the deployable `residual-delta` router properly and take it to tier 2; the confound is not the obstacle |
| T10 P10.3 (λ* far from 1) | Report MDL as a regulariser and drop the principle framing from T11; do **not** re-derive the accountant to rescue λ=1 |
| T11 P11.1 (monotone in `p`) | Length axis: seq ∈ {128, 512, 2048} at the two ends of `p` only, tier 2. Three points, one claim |
| T12 Gate A (`R²` < 0.5) | Stop. Publish the probe as a negative result about the premise |

---

## 6. Work breakdown

| # | item | depends on | est. |
|---|---|---|---|
| 1 | `suite/budget.py` + tests | — | 0.5 d |
| 2 | `suite/coding.py` (arithmetic coder + round-trip test) | — | 1 d |
| 3 | `suite/preregister.py` | — | 0.25 d |
| 4 | **T6** — metrology gate | 2, 3 | 0.5 d + 2.0 h |
| 5 | `suite/routing.py` (router ABC, capacity contract, causality assert) | 1 | 1 d |
| 6 | **T7** — conditional width | 1, 3, 4 | 0.5 d + 21.9 h |
| 7 | MoD/MoR blocks in `hybrid.py` | 5 | 1 d |
| 8 | **T8** — conditional depth | 6, 7 | 0.5 d + 5.4 h |
| 9 | **T9** — routing signal (incl. offline oracles) | 8 | 1 d + 3.1 h |
| 10 | `suite/mdl.py` (two-part accountant, hard-concrete gates) | 2 | 1.5 d |
| 11 | **T10** — MDL objective | 10 | 0.5 d + 6.5 h |
| 12 | `suite/graphs.py` (masks + graph metrics) | — | 1 d |
| 13 | Synthetic retrieval tasks (needle, multi-hop) | — | 0.5 d |
| 14 | **T11** — topology | 11, 12, 13 | 0.5 d + 12.9 h |
| 15 | **T12-A** — redundancy probe | 6 | 0.25 d + 1.5 h |
| 16 | **T12-B** — one-pass residual coding | 15 | 1 d + 3.5 h |
| 17 | **T12-C** — full predictive coding | 16 | 1 d + 3.6 h |

≈ 13 engineering days, **≈ 63 CPU-hours nominal / ~81 h with contingency**
(§4.4.1), with four independently runnable branches. Item 1a — vectorise
`MoE.forward` — is not in the list above and should be: one day of work that
removes ~15 h of compute and makes P7.4 answerable.

---

## 7. What would falsify the framing itself

The §0 chain — compression / entropy / energy / sparsity-as-consequence — makes
predictions that span experiments, and it is worth stating what would sink it
rather than any individual arm:

* **Sparsity helps where no constraint binds.** If T11's masked models beat dense
  attention at *equal* compute with no budget pressure, sparsity is doing
  something other than satisfying a constraint, and the solution-side story is
  wrong.
* **The budget constraint does not produce vertex solutions.** If T8's learned
  allocation is near-uniform across tokens at every capacity `c` — with
  `load_balance` near 1 and selection entropy near maximal — then the constrained
  optimum is interior, not a vertex, and §0's step 4 does not describe what
  training finds.
* **Pricing structure does not find structure.** If T10's `l0-structural` prunes
  indistinguishably from `l0-shuffled`, then charging for description length is
  not selecting *for* compressible structure, and the data-side story has no
  support here.
* **The two stories never separate.** If T7/T8 and T10/T11 always agree, the
  suite cannot tell solution-side from data-side sparsity, and its central
  organising distinction is not measurable at this scale. That is a failure of
  the *design*, and it would be the most important thing this suite could learn
  about itself.
