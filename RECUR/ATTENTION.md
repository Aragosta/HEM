# Round three: the attention temperature

Three experiments on one scalar, and they produced both the largest effect in
this project and the demotion of its previous headline result.

**Why this round exists.** T2b measured that the loop is a contraction: expert
overlap rises (0.49 → 0.71 → 0.85) exactly as the state update falls
(0.51 → 0.19 → 0.08), r = −0.877. Routing is a linear map of the state
(arXiv:2604.09780), so routing convergence was a *symptom*. The state is written
by attention, and a head computes `softmax(β · q·k)` — a Boltzmann distribution
over keys at inverse temperature β = 1/√head_dim, a constant nobody chose as a
temperature. `../CALM/CRITICALITY.md` §1 collects the theory: β has a phase
diagram, with a disordered phase (uniform attention, rank collapse), a frozen
phase (hard argmax, copies one token) and a useful regime in between.

Nothing in the first 146 runs had measured where our model sits on that axis.

> **A note on the word "temperature".** β is the *inverse* temperature:
> `p ∝ exp(−βE)`. High β means **low** temperature — sharp, frozen attention.
> Low β means **high** temperature — soft, uniform attention. Earlier drafts of
> this file used "colder" for lower β, which is backwards. Every number below is
> unchanged; the optimum is at **lower β**, i.e. **softer, hotter** attention
> than the `1/√d` default, and the frozen phase is the *cold* one.


---

## The verdict, against the predictions registered before the runs

| # | prediction | outcome |
| --- | --- | --- |
| A1a | accuracy has an interior optimum in β | **held, emphatically** — a 16× sweep moves accuracy from 0.709 to 0.447 against a seed spread of 0.004–0.046. This is the largest effect measured anywhere in this project |
| A1b | attention entropy falls across loops | **held, with the same shape as everything else** — sharp drop from loop 1 to loop 2, then flat (0.711 → 0.669 → 0.678 → 0.672) |
| A1c | the optimum is *hotter* than the default | **failed, in the opposite direction** — the optimum is at **lower β — softer attention** (×0.5 at R=1, ×0.25 at R=4). I predicted from the entropy of an *untrained* model; training sharpens attention on its own, so the default already sits on the sharp (cold) side and the useful correction is to soften |
| A2a | a learned per-loop β beats a fixed one | **failed** — −0.030 against doing nothing |
| A2b | temperature and routing gains do not add | **held, and then some** — they *interfere*: +0.062 (routing) and −0.030 (temperature) combine to +0.021, below either the sum or the larger single |
| A3a | the router bias adds nothing at the right temperature | **held** — −0.018. At ×0.25 the plain model scores 0.678; adding E3's per-loop bias gives 0.660 |
| A3b | the two are different mechanisms and both belong | **failed** — see A3a |

**The unpredicted finding, and the one worth carrying forward: the optimal β
moves with loop count.** Best is ×0.5 at R=1 and ×0.25 at R=4 — *softer* as
depth grows. A weight-shared head run repeatedly over a contracting state
sharpens itself with every pass, so it has to *start softer* to stay in the
useful regime — which is the "the loop walks along the β
axis" story showing up as a measurement rather than an argument.

**And the demotion.** Loop-index-conditioned MoE routing was round one's only
result that repeated (+0.037 in E3, +0.063 in T2b). A3 shows it is substitutable
by one scalar: temperature alone buys +0.070, more than the router bias's
+0.062, and stacking them buys nothing. Two routes to the same gain, and the
cheaper one — 16 parameters against 64 — is the larger. E3's result was real but
it was not about experts.

---

## 1. A1 — the temperature sweep

<!-- filled by: python a1_temperature.py --report-only -->

## accuracy against beta (x 1/sqrt(head_dim))

| loops | x0.25 | x0.5 | x1 | x2 | x4 | best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.693 ± 0.044 | 0.709 ± 0.018 | 0.692 ± 0.037 | 0.561 ± 0.017 | 0.447 ± 0.004 | **x0.5** |
| 4 | 0.678 ± 0.046 | 0.639 ± 0.010 | 0.608 ± 0.036 | 0.607 ± 0.032 | 0.456 ± 0.006 | **x0.25** |

## the order parameter: normalised attention entropy

1.0 = uniform attention (disordered); 0.0 = hard argmax (frozen).

| loops | beta | entropy per loop | final |
| --- | --- | --- | --- |
| 1 | x0.25 | 0.744 | 0.744 |
| 1 | x0.5 | 0.478 | 0.478 |
| 1 | x1 | 0.414 | 0.414 |
| 1 | x2 | 0.155 | 0.155 |
| 1 | x4 | 0.058 | 0.058 |
| 4 | x0.25 | 0.711, 0.669, 0.678, 0.672 | 0.672 |
| 4 | x0.5 | 0.548, 0.447, 0.440, 0.443 | 0.443 |
| 4 | x1 | 0.579, 0.520, 0.532, 0.532 | 0.532 |
| 4 | x2 | 0.202, 0.190, 0.189, 0.187 | 0.187 |
| 4 | x4 | 0.087, 0.091, 0.097, 0.100 | 0.100 |

## 2. A2 — per-loop temperature against per-loop routing

<!-- filled by: python a2_beta_vs_router.py --report-only -->

| arm | accuracy | sd | n | vs none | params |
| --- | --- | --- | --- | --- | --- |
| none | 0.608 | 0.036 | 2 | - | 278,080 |
| beta | 0.578 | 0.017 | 2 | -0.030 | 278,144 |
| bias | 0.671 | 0.032 | 2 | +0.062 | 278,336 |
| both | 0.629 | 0.039 | 2 | +0.021 | 278,400 |

## additivity (A2b)

- temperature alone: -0.030
- routing alone: +0.062
- both together: +0.021
- sum of the two singles: +0.032
- larger single: +0.062

Closer to the larger single than to the sum means the two are one intervention applied at two points in the same pipeline; closer to the sum means they are different mechanisms and the upstream story is wrong.

## what the temperature learned, per loop

| arm | seed | entropy per loop |
| --- | --- | --- |
| none | 0 | 0.579, 0.520, 0.532, 0.532 |
| none | 1 | 0.403, 0.356, 0.354, 0.356 |
| beta | 0 | 0.425, 0.299, 0.301, 0.299 |
| beta | 1 | 0.470, 0.460, 0.499, 0.511 |
| bias | 0 | 0.395, 0.350, 0.354, 0.358 |
| bias | 1 | 0.529, 0.457, 0.469, 0.462 |
| both | 0 | 0.406, 0.322, 0.355, 0.339 |
| both | 1 | 0.531, 0.443, 0.490, 0.488 |

## 3. A3 — does the router bias survive the right temperature?

<!-- filled by: python a3_cold_plus_router.py --report-only -->

| arm | accuracy | sd | n |
| --- | --- | --- | --- |
| cold (x0.25, no bias) | 0.678 | 0.046 | 2 |
| cold + per-loop bias | 0.660 | 0.004 | 2 |
| default temperature, no bias | 0.608 | 0.036 | 2 |
| default temperature + bias | 0.671 | 0.032 | 2 |

- router bias at the default temperature: +0.062
- router bias at the cold temperature: -0.018
- temperature alone (x0.25 vs x1, no bias): +0.070

## What this changes about the whole project

Reading the three rounds together:

1. **Depth (E1)**: worth about two loops, and a substitute for data rather than
   a complement. The ceiling does not move with the token budget.
2. **The mechanism (T2b)**: the loop is a contraction; routing convergence is
   the shadow of state convergence, r = −0.877.
3. **The knob (A1–A3)**: the contraction is governed by attention temperature,
   the optimum moves to softer attention as depth grows, and getting it right subsumes the
   only architectural win the earlier rounds produced.

That is a coherent story, and it points somewhere unglamorous: before adding
registers, gates, expert conditioners or depth-wise residuals to a looped model,
**tune one scalar per head, per loop position**, and check whether the thing you
were about to build still buys anything.

What would falsify it: a model where β is tuned and the router bias *still*
helps. A3 says that does not happen at 0.3M parameters on a 6-entity
composition task with sequences of 37 tokens. Whether it holds where attention
has real work to do — long context, many heads, a vocabulary worth attending
over — is exactly the untested part, and the reason the critical scaling in
`CRITICALITY.md` is stated as β ≍ log n rather than as a constant.
