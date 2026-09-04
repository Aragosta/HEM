# Optimal graph reduction, criticality, and what physics-of-the-brain
# actually offers a transformer

Question asked: *is there a way starting from dense, going to sparse, optimizing
some objective, that could allow for critical phase transitions? Or other
physics ideas of the brain?*

Short answer: **yes, and a transformer is already doing it** — softmax attention
is a dense graph continuously sparsified by a temperature parameter, that
parameter has a *known* critical scaling, and the phase diagram has three
regimes with the useful one in the middle. The literature that names this is
spread across four disconnected communities (statistical physics of attention,
GNN rewiring, network pruning, neuroscience criticality) which mostly do not
cite each other. This document connects them and extracts what is actually
testable here.

Everything below is a literature claim with its source; the "What this repo can
test" sections are mine and are not established results.

---

## 1. Attention is a dense graph with a temperature knob, and it has a phase transition

The single most relevant result:

**Critical attention scaling in long-context transformers** (Chen, Lin,
Polyanskiy, Rigollet — `arxiv:2510.05554`). Treat one attention head over `n`
tokens as a weighted complete graph whose edge weights are `softmax(β·qᵀk)`.
`β` is an inverse temperature (in a standard transformer, `β = 1/√d_head`).
As `n → ∞` the head has a phase transition in `β`:

| regime | behaviour |
|---|---|
| `β` too small | **disorder / rank collapse.** All queries attend near-uniformly; every output token converges to the same direction. The head computes a mean and carries no information. |
| `β` critical, `β_n ≍ log n` | **the interesting phase.** Weight concentrates on a small, content-dependent subset of tokens. |
| `β` too large | **frozen order.** Attention becomes the identity operator (or a hard argmax); the head copies one token and stops mixing. |

The critical scaling is `β_n ≍ log n`. This is a *derivation* of what YaRN,
Qwen's length-scaling, and SSMax all do empirically: they rescale attention
logits by something like `log n` when context grows. Without it, a head trained
at 2k tokens is pushed into the disordered phase at 128k — not because of
positional encoding, because of thermodynamics.

Related: `arxiv:2605.08505` gives a critical scale `β_n* ~ n^{2/(d−1)}` in a
different asymptotic regime — note the **explicit dependence on head dimension
`d`**. `arxiv:2303.06296` shows attention-entropy collapse coincides with
training instability (entropy is exactly the order parameter you would pick).
`arxiv:2601.19942` gives an order parameter `Ω(h) = 1 − ‖h‖₁/(√d‖h‖₂)` that is
discontinuous at a critical depth. `arxiv:2512.01868` treats attention as an
interacting particle system with Kuramoto-style synchronisation.
`arxiv:2606.12058` finds copy-head emergence is itself a phase transition.

**The reduction is already happening.** Softmax does not select edges, it
anneals them. "Sparsifying attention" and "operating at criticality" are the
same operation viewed from two disciplines.

### What this repo can test

The T1 sweep currently running varies `head_dim ∈ {16, 32, 64}` at fixed total
width 192. Since `β = 1/√d_head`, that sweep is *also* a sweep of inverse
temperature — 0.25, 0.177, 0.125. It was designed to test dimension efficiency
of hyperbolic heads; it doubles as a coarse scan of the `β` axis. A follow-up
that decouples them (fix `head_dim`, learn or schedule a per-head `β`, and log
attention entropy per layer) would separate "geometry helps" from "temperature
was mis-set", which the current design cannot.

---

## 2. Curvature is the right objective for graph reduction — and it is the same
## curvature the hyperbolic work is about

This is the bridge between the two threads in this repo.

**Understanding over-squashing and bottlenecks on graphs via curvature**
(Topping et al., `arxiv:2111.14522`) and **Revisiting Over-smoothing and
Over-squashing Using Ollivier-Ricci Curvature** (`arxiv:2211.15779`) establish:

- **negatively curved edges cause over-squashing** — information from an
  exponentially growing neighbourhood is forced through a bottleneck;
- **positively curved edges cause over-smoothing** — neighbourhoods overlap, so
  repeated averaging washes out distinctions.

So discrete Ricci curvature is a *local, computable, signed* diagnostic that says
which edges to add and which to delete, and the two failure modes sit at opposite
signs. That makes it a genuine objective for dense→sparse reduction rather than a
heuristic. Follow-ups: Forman-Ricci augmentations (`arxiv:2309.09384`), entropic
/ global transport curvature (`arxiv:2607.22381`), physics-informed Ollivier-Ricci
flow (`arxiv:2504.04052`), and a skeptical replication finding much of the
reported gain is hyperparameter noise (`arxiv:2407.09381` — worth taking
seriously before building on this).

**The connection to HELM.** Hyperbolic space is uniformly negatively curved.
Embedding a hierarchy in H^d is the *dual* move to curvature rewiring: instead of
editing the graph so that message passing survives a bad geometry, you change
the geometry so the graph's native tree-likeness is no longer a bottleneck. The
FlyWire result in `WHY_HYPERBOLIC.md` (2D hyperbolic greedy routing 0.553 vs the
fly's own 3D anatomy at 0.075) is exactly a statement about over-squashing:
negative curvature makes greedy routing work at a dimension where Euclidean
routing does not. And the crossover finding there — Euclidean Node2vec overtakes
between d=8 and d=16 — says the advantage is dimension efficiency, which is why
the sensible place to put it is the *per-head* space (64–128 dims) and the KV
latent, not the residual stream.

### What this repo can test

Compute Ollivier-Ricci curvature of the attention graph, per layer, per head,
for the euclidean and lorentz arms of T1. Prediction worth registering *before*
looking: if hyperbolic heads help at all, their attention graphs should show a
**less negative** curvature tail than Euclidean heads at the same `head_dim` —
the geometry has absorbed the bottleneck. If the curvature distributions are
indistinguishable, the geometry is decorative and the perplexity numbers (which
so far say Euclidean wins) are the whole story.

---

## 3. Deleting edges can *improve* flow: the Braess phenomenon

**Spectral Graph Pruning Against Over-Squashing and Over-Smoothing**
(`arxiv:2404.04612`) is the counterintuitive one. Standard intuition says adding
edges fixes over-squashing but worsens over-smoothing, so the two are in
tension. This paper invokes the **Braess paradox** — the traffic result where
*closing* a road can reduce everyone's travel time — and shows that in spectral
terms, edge *deletion* can improve both simultaneously. Reduction is not a cost
paid for efficiency; there is a regime where it is strictly better.

Supporting machinery: effective-resistance spectral sparsification provably
preserves what a graph attention layer computes (`arxiv:2006.08796`); expander
graphs give a principled sparse attention pattern with a guaranteed spectral gap
(**Exphormer**, `arxiv:2303.06147`, and `arxiv:2411.16278`, which observes that
learned attention already uses very few of the available edges); graph lottery
tickets (`arxiv:2102.06790`, `2305.02190`, `2402.01261`).

The objective, then, is not "keep the most weight" but **preserve the spectral
gap / Cheeger constant while removing edges**. That is a well-posed optimisation
with Cheeger inequalities as the guarantee.

---

## 4. Pruning in neural networks is a genuine phase transition

Not a metaphor — measured with the apparatus of statistical mechanics:

- **Phase Transitions in Neural Networks Pruning** (`arxiv:2602.15224`): magnitude
  pruning with fine-tuning shows a *sharp* transition from a functional to a
  degraded phase, analysed as a critical phenomenon rather than a smooth
  degradation curve.
- **Pruning-induced phases in fully-connected networks** (`arxiv:2603.12316`):
  maps a phase diagram in (train-time dropout, eval-time dropout) and asks the
  universality-class question directly.
- **A Three-regime Model of Network Pruning** (`arxiv:2305.18383`): temperature-like
  and load-like control parameters from the statistical mechanics of learning
  predict prunability from training hyperparameters.
- **Optimal pruning in neural networks** (`pmid:11138138`): the optimal pruning
  threshold grows as `[ρ₀ − ρ_c(κ)]^{1/2}` above a critical value — a **critical
  exponent of 1/2**, the mean-field value. This is a 2000-era result and it is
  the cleanest statement in the whole area.
- **A Tale of Two Circuits** (`arxiv:2303.11873`): grokking *is* a competition
  between a dense and a sparse subnetwork, with the transition being the sparse
  one taking over. Dense→sparse is the mechanism of a known generalisation
  phase transition.

Note the direction of the finding: there is a **critical sparsity** below which
nothing degrades and above which everything does. The engineering goal is to sit
just under it, which is the same "edge of chaos" statement as §5, arrived at from
weights instead of activity.

---

## 5. The brain's version: criticality reached by purely local rewiring

The neuroscience thread supplies the mechanism the ML thread lacks — how a
network *finds* the critical point without being told where it is.

- **Self-organized criticality in neural networks from activity-based rewiring**
  (`arxiv:2009.11781` / `pmid:33862737`): a minimal model where incoming links are
  added or deleted based only on a node's own average activity. No global signal,
  no target sparsity — the network self-organises onto the critical point of a
  dynamical phase transition. **This is the important one:** it is a local
  add/delete rule that solves a global optimisation implicitly.
- **Analytical investigation of SOC in neural networks** (`arxiv:1203.4942`):
  proves that homeostatic-plasticity-inspired rewiring creates an *attractive*
  steady state at criticality — a transcritical bifurcation in the macroscopic
  dynamics. The critical point is a fixed point of the rewiring rule, not a
  coincidence.
- **Adaptive self-organized criticality in deep neural networks**
  (`arxiv:2608.28431`): the same idea imported into deep nets — purely local
  homeostatic plasticity regulates the *global* dynamical state of a deep
  network, using only pre- and post-synaptic activity.
- **Growth strategy determines network performance** (`arxiv:1806.01878`): models
  synaptic pruning with coupled activity and topology, reproducing the measured
  developmental profile of synaptic density, and proves the **initial transient
  of high connectivity is necessary** for the ordered stationary states that can
  store stable memories. Dense-then-prune beats grow-sparse — not as an
  optimisation convenience, as a structural requirement.
- Criticality yields optimal representations (`arxiv:2307.10669`), optimal input
  representation at the edge of chaos (`PMC8389338`), whole-brain crackling noise
  (`PMC6307982`), E/I balance and clustering (`arxiv:2202.03330`).

`arxiv:1806.01878` and `arxiv:2603.12316` are saying the same thing from
biology and physics: **over-parameterise, then reduce**, and the reduction has a
critical point you want to approach but not cross.

---

## 6. Synthesis — what is actually on offer

Stacking the four literatures:

1. A transformer layer is a **dense graph** (attention) plus a **discrete graph
   reduction** (MoE top-k routing — which is exactly a hard, learned, sparse
   edge selection over the expert graph, and nobody in the criticality
   literature seems to have noticed).
2. The dense graph is already continuously sparsified by `β`, with a **known
   critical scaling** `β_n ≍ log n` and three phases, the useful one being an
   intermediate regime of content-adaptive sparsity (§1).
3. There is a **principled objective** for reduction — preserve the spectral gap,
   fix the sign of the Ricci curvature — and a result that deletion can strictly
   improve information flow (§2, §3).
4. There is a **mechanism for finding the critical point without knowing it**:
   local activity-based add/delete rules provably have criticality as an
   attractive fixed point (§5).
5. And **curvature is the shared variable** between the reduction question and
   the hyperbolic question. Hyperbolic embedding and curvature-based rewiring are
   two ways to remove the same bottleneck.

The honest caveat: (2) is asymptotic theory about a single head, (3) is
demonstrated on GNN benchmarks whose gains have been partly disputed
(`arxiv:2407.09381`), (4) is on small dynamical models, not transformers, and (5)
is my synthesis, not a published result. None of this is evidence that a
criticality-aware transformer beats a well-tuned baseline. It is a set of
well-posed hypotheses with measurable order parameters, which is more than most
architectural intuition has.

## 7. Concrete next experiments, cheapest first

Ordered by cost, each falsifiable, each with a registered prediction:

> **Implementation status.** C1 and C2 are built (`suite/hybrid.py`,
> `suite/t2_criticality.py`): `beta_mode` makes the attention temperature a
> per-head learnable scalar initialised so that step 0 is bit-identical to the
> standard `1/sqrt(d)`, and `set_stats(True)` records the order parameters.
> Measured at initialisation they read `entropy_norm = 1.0000`,
> `participation_frac = 0.9997` -- a random model sits fully in the disordered
> phase, as the theory requires. One trap found while building it: `AdamW`'s
> weight decay applied to `log_beta` pulls it toward 0, i.e. beta toward 1.0,
> which from a base of 0.25 is a 4x sharpening produced by the optimizer. That
> would have made C2's prediction answer itself; decay is now restricted to
> `ndim >= 2`. C3-C5 are not built.

- **C1 — measure the order parameter (near-free).** Log per-layer, per-head
  attention entropy and the `Ω(h)` order parameter across the existing T0/T1
  runs. *Prediction:* the failing HELM arm from T0 (269 vs 86 perplexity) sits in
  the **disordered** phase — low `β` effective, high entropy, rank collapse —
  because the Lorentz inner product's magnitude scale differs from the Euclidean
  one and `1/√d` is then the wrong temperature. If true, T0's result is a
  temperature bug, not a verdict on geometry, and is fixable with one scalar.
- **C2 — sweep `β` explicitly (one small run).** Fix `head_dim`, scan a learnable
  or scheduled logit scale. *Prediction:* both geometries have an optimum, and the
  Lorentz optimum sits at a different `β` than the Euclidean one. This is the
  cheapest test that could overturn the T0 conclusion, and it should be run
  before any more geometry work.
- **C3 — curvature of the attention graph.** Ollivier-Ricci on the learned
  attention graphs from the T1 arms, per §2.
- **C4 — local rewiring on MoE routing.** Replace fixed top-k with an
  activity-based add/delete rule over the expert graph (§5), targeting a measured
  criticality statistic rather than a fixed k. *Prediction:* matches top-k at
  equal average sparsity but is more robust to the expert-collapse failure mode.
- **C5 — dense-then-prune schedule.** Train dense, prune toward the critical
  sparsity of §4. *Prediction:* per `arxiv:1806.01878`, the dense transient is
  necessary; a matched-FLOP always-sparse run underperforms.

C1 and C2 are the ones that matter now, because if C1's prediction holds then the
central negative result in `RESULTS.md` is measuring the wrong thing.
