# Physics and neuroscience for a Kimi-shaped model

The question behind this document: *can criticality, statistical physics, and
the physics of the brain be used to improve a frontier MoE architecture, using
Kimi K3 (`arxiv:2607.24653`) as the concrete target?*

Short answer: **yes, but not where `CRITICALITY.md` was aiming.** The attention
softmax is the wrong object — its spectrum is dominated by an algebraic
artefact, and its temperature is redundant with the weights that produce the
logits (§5). The right object is the **linear-attention state recurrence**,
which is a genuine dynamical system with a genuine spectrum, and which K3 now
runs in 69 of its 93 layers. Everything the criticality literature knows how to
say — spectral radius, Lyapunov exponents, memory curves, non-normal transient
amplification — is well posed there and ill posed for softmax.

Written after reading the "Learning and pruning a network toward criticality"
research synthesis (4 September 2026), which is the direct source of §5.1's
correction and of the taxonomy in §1.

---

## 0. What K3 actually is, in the terms of this repo

| axis | K3 mechanism | what it is, dynamically |
|---|---|---|
| sequence | 3 KDA layers : 1 Gated MLA layer per block | a gated delta-rule recurrence, punctuated by full attention |
| depth | AttnRes — learned `α` over embedding and all preceding block outputs | soft, learned routing across depth |
| width | Stable LatentMoE — 16 of 896 routed experts, 2 shared, in a half-width latent | a hard, learned graph reduction, re-decided per token |
| stability | RMSNorm before the up-projection, SiTU-GLU, Quantile Balancing, Per-Head Muon | gain control and homeostasis, by construction |

The KDA state update, which is the object §2 is about:

    S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ,     ‖k_t‖ = 1

`α_t ∈ (0,1)^{d_k}` is a channel-wise retention factor produced from a low-rank
decay logit through what the paper calls a *lower-bounded mapping* — a hand-set
floor. `β_t ∈ (0,1)` is the delta-rule write strength. `q` and `k` are
L2-normalised after a short convolution and Swish.

Three observations follow immediately, and they are the whole of this document's
positive content.

**(a) There is no Perron artefact here.** For softmax attention `P` is
row-stochastic, so `P1 = 1` and an eigenvalue of exactly one exists regardless
of training; reading criticality off it measures the normalisation, not the
model. `A_t = (I − β_t k_tk_tᵀ)Diag(α_t)` carries no such constraint. Its
spectral radius, its singular values, and finite-time Lyapunov exponents of the
`A_t` product are all meaningful and cheap.

**(b) K3 already sets a criticality band, by hand.** The lower bound on `α` is a
floor on the spectral radius of the recurrence: retain at least this much, or
long-range memory dies. It is a threshold chosen by taste. This is exactly the
quantity for which physics has quantitative theory (§2).

**(c) The delta term is what makes the operator non-normal.** `Diag(α)` alone
(the Mamba-style gated-decay form) is normal. `(I − βkkᵀ)` is a rank-one
correction that breaks normality. Ganguli, Huh & Sompolinsky (`pmcid:PMC2596211`)
proved that normal recurrent networks have `O(1)` memory traces while non-normal
ones can have *extensive* ones. So the mechanism by which a delta rule beats a
pure decay gate has a name in physics, and a measurable signature.

---

## 1. Five things called "criticality", and which one applies where

Taken from the September research synthesis; kept here because conflating these
is the standard failure mode of this literature.

| regime | order parameter | target | applies to |
|---|---|---|---|
| avalanche / branching | branching ratio `m` | `m ≈ 1` | event-driven recurrent nets; **not** a feed-forward transformer |
| excitable graph | spectral radius of weighted adjacency | `ρ ≈ 1` | **the KDA recurrence** |
| edge of chaos | max finite-time Lyapunov exponent | `λ ≈ 0` | **the KDA recurrence**; looped transformers |
| depth signal propagation | token-angle / gradient growth exponents | near-zero | the residual stack across depth |
| optimizer edge of stability | top Hessian eigenvalue | `≈ 2/η` | parameter updates; says nothing about activations |

The KDA layers are the only place in a K3-shaped model where rows two and three
are literally true rather than analogies. That is the reason to attack there.

---

## 2. The strongest thread: the spectrum of the state recurrence

### 2.1 Where the band belongs is a solved problem in physics

- **Fisher Memory Curve** (Ganguli, Huh & Sompolinsky, `pmcid:PMC2596211`).
  Gives the signal-to-noise ratio a linear recurrent state retains about an
  input `k` steps back, as a function of the network operator. The headline:
  memory is `O(1)` for normal operators and **extensive** for non-normal ones.
  KDA is non-normal by construction; nobody has measured its FMC.
- **Optimal short-term memory sits *before* the edge of chaos**
  (`arxiv:1912.11213`, `pmid:31962477`). Mean-field derivation of memory
  capacity, mutual information and Fisher information for driven random
  recurrent networks: the optimum is slightly *sub*critical. This matches the
  neuroscience finding that awake cortex is "reverberating" — branching ratio
  below but close to one — and it argues for a **band**, not `ρ = 1`.
- **Edge-of-stability echo state networks** (`arxiv:2308.02902`) and Fisher-
  information determination of the critical point (`arxiv:1603.03685`,
  `pmid:28092580`) give the reservoir-computing version, including a practical
  method for locating the edge from data rather than assuming it.
- **Gating creates slow modes** (`arxiv:2002.00025`, and the full theory in
  `pmcid:PMC9762509`). The physics of what a forget gate does to phase-space
  structure. This is the theoretical account of `α_t`.

### 2.2 Three recent results that bear directly on KDA

- **Variational Linear Attention** (`arxiv:2605.11196`) proves that normalising
  the write direction to unit length makes the recurrence Jacobian's spectral
  norm **exactly 1** for every sequence length and head dimension. That is
  criticality by construction — a designed competitor to K3's hand-set `α`
  floor, with a proof where K3 has a threshold.
- **SANE** (`arxiv:2608.22354`) tracks RWKV-7 to 100M-token contexts and finds
  the failure mode is *localised* norm explosion in a few channels over a sparse
  substrate, not global saturation. A per-channel criticality failure — exactly
  what a channel-wise spectral band would catch and what any scalar diagnostic
  would miss. K3's `α` is channel-wise, so the diagnostic and the control knob
  are already at the same granularity.
- **LayerNorm as implicit gain control in looped transformers**
  (`arxiv:2607.10681`). Under non-normality the operative quantity is the
  *spectral margin*, not an operator-norm bound, and that margin depletes as the
  carry `ρ → 1`; `ρ(A) < 1` is necessary but not sufficient. This is the
  synthesis report's "use singular gains, not eigenvalues" turned into a
  quantitative statement about the architecture family K3 uses.

Adjacent design work worth reading before proposing anything: Gated DeltaNet-2
(`arxiv:2605.22791`, decouples erase from write on the grounds that KDA's single
`β` conflates two jobs), OSDN (`arxiv:2605.13473`, per-feature preconditioning of
the delta step), Kaczmarz Linear Attention (`arxiv:2605.08587`, derives the
gating coefficient instead of learning it), and Exact Flow Linear Attention
(`arxiv:2512.12602`, the exact continuous-time flow whose Euler discretisation is
the delta rule).

### 2.3 The claim worth testing

> KDA's advantage over pure gated decay is a **non-normality effect**, visible
> as an extended Fisher memory curve, and the useful design lever is the gap
> between `‖A_t‖` and `ρ(A_t)` — transient amplification — rather than the
> height of the `α` floor.

Falsified if the FMC of a KDA layer is indistinguishable from a `Diag(α)`-only
layer at matched `ρ`, or if recall performance tracks `ρ` alone.

---

## 3. The router as a homeostatic controller

Quantile Balancing is a control law over ~10³ units whose stated purpose is
preventing dead experts at a sparsity where auxiliary-loss-free bias updates
stop behaving. Structurally that is homeostatic regulation of excitability, and
the neuroscience has the theory the engineering lacks:

- `arxiv:1203.4942` proves homeostatic-plasticity-inspired rewiring makes
  criticality an **attractive** fixed point — a transcritical bifurcation in the
  macroscopic dynamics, not a coincidence of tuning.
- `arxiv:2009.11781` / `pmid:33862737` reach the critical point with a purely
  *local* add/delete rule driven by a node's own average activity: no global
  signal, no target sparsity.
- `arxiv:2608.28431` imports the same idea into deep networks: local homeostatic
  plasticity regulating a global dynamical state.
- Zeraati, Priesemann & Levina's review is the map of the mechanism space —
  short-term plasticity as fast negative feedback, long-term rules for a
  persistent set point, and the requirement that Hebbian learning be paired with
  normalisation or it runs away.

**The design this suggests.** Give each expert its own excitability threshold,
regulated by its own recent load, and drop fixed top-k: `k` becomes per-token
variable and the model buys a Pareto curve instead of a point. This is C4 in
`CRITICALITY.md`, and the part that is still novel post-`arxiv:2605.06415` is
**measuring the order parameter**, not balancing the load.

What already exists, and must be baselined against:

| work | what it does |
|---|---|
| `arxiv:2408.15664` | DeepSeek's auxiliary-loss-free balancing — the bias-update rule K3 descends from |
| `arxiv:2512.03915` | that rule as a primal-dual method, with theory |
| `arxiv:2605.06415` | `E = T·H/(O+B)`: a dimensionless control parameter, with `E ≥ 0.5` sufficient for zero dead experts across 12 experiments — a phase diagram for expert ecology |
| `arxiv:2605.17598` | deep-layer routing collapse measured by usage entropy in production MoEs |
| `arxiv:2605.15403` | φ-balancing: population-level balance by convex duality |
| `arxiv:2605.00604` | free-energy MoE — per-expert LIF membrane potential, precision-weighted gating; the closest existing bio-inspired router, small and controlled |
| `arxiv:2507.03221` | neural inhibition improves dynamic routing |

---

## 4. Two cheaper threads

### 4.1 The shared/routed ratio is an E/I-balance question

`arxiv:2007.02511` / `pmcid:PMC8962757`: critical avalanches arising from
excitation–inhibition balance under **modular** topology achieve lower wiring
*and* firing cost with strongly enhanced stimulus sensitivity. That is the
theory-side analogue of the shared-plus-routed split — shared experts are the
dense always-on population, routed experts the sparse specialists. K3 moved
shared experts 1 → 2 while routed went 384 → 896, with no stated principle for
the ratio. `suite/t4_expert_ratio.py` already sweeps this axis; E/I balance turns
that sweep into a prediction.

### 4.2 Attention sinks are self-organised homeostasis

The sink literature converged in the last year and nobody has framed it in
criticality terms:

- `arxiv:2601.22966` — sinks and residual outliers perform *outlier-driven
  rescaling*, and it is **essential** for training, not a pathology.
- `arxiv:2510.06477` — sinks and compression valleys are two faces of massive
  activations, with proven bounds on the entropy reduction.
- `arxiv:2603.17771` — sinks induce *gradient* sinks: massive activations act as
  gradient regulators through the RMSNorm Jacobian.
- `arxiv:2504.20966` (softpick) — a rectified, non-sum-to-one softmax removes
  sinks and massive activations entirely, at 0% sink rate.
- `arxiv:2410.13835`, `arxiv:2603.05498`, `arxiv:2605.06611` — mechanism and
  structural origin.

Read together: a standard transformer manufactures a pressure-release valve to
hold its own gain inside a band. That is the best existing counter-evidence to
the claim that transformers do not self-organise their dynamical state — and the
T5 `sigmoid` arm is already a non-simplex normaliser in the softpick family, so
the prediction is free (§6, T6).

### 4.3 General file

`arxiv:2606.10384` (branching ratio ≈ 1 in small LSTMs near their optimal
training step, subcritical in larger ones — a direct warning about whether this
suite's scale can see the effect at all), `arxiv:2509.22649` (avalanche equations
from non-equilibrium statistical physics applied to DNN cascades: networks learn
best poised between absorbing and active phases), `arxiv:2203.12967` (extended
critical regimes from heavy-tailed weights), `arxiv:2507.08527`, `arxiv:2604.16431`
(avalanche probe of grokking).

---

## 5. Mistakes found while doing this

Recorded because the corrections change what existing results mean. The first
two are patched in `CRITICALITY.md`.

**5.1 T2's `β` null is confounded, and the fix is three lines.** *Tested in
T2-redux; the conclusion holds and the mechanism given below does not — see
`suite/RESULTS.md`. With QK-norm the learned `β` settles at 1.110 ± 0.043
against 0.986 ± 0.009 without it, complete separation across four seeds, so the
null was indeed an artefact of the parameterisation. But the paired q/k gain
difference is +0.755 ± 7.236 with 2 of 4 signs positive: the endpoint
compensation asserted below is **not observed**, and is withdrawn. What the data
supports is the weaker claim that a redundant direction receives no consistent
gradient, so `β` never moves far — it does not need the weights to chase it.* `hybrid.py`
takes `q` and `k` straight from their linear projections (RoPE only, **no
QK-norm**), so the effective inverse temperature is `‖q‖·‖k‖·β`. A learnable
per-head `β` is then an exactly redundant copy of the q/k gain, and `RESULTS.md`
measured the compensation happening: `β` settled 7% *flatter* while the final
`participation_frac` came out *lower*. So T2 established that a redundant
parameterisation buys nothing, which is nearly a tautology — not that
temperature is inert. The decisive version applies QK-norm, which deletes the
compensating path. Note that K3 L2-normalises `q` and `k` inside KDA, and K2
needed MuonClip/QK-clip, precisely because logit scale is otherwise an
uncontrolled variable.

**5.2 The T1 head_dim sweep is not a `β` sweep.** At fixed width it moves `β`,
the per-head subspace dimension, and the head count together; and the critical
scaling in `arxiv:2510.05554` is in the **context length** `n` (`β_n ≍ log n`),
which T1 holds fixed at 128 in every arm.

**5.3 One citation dropped.** The order parameter `Ω(h) = 1 − ‖h‖₁/(√d‖h‖₂)` is
the Hoyer sparsity measure (2004) and a monotone reparameterisation of the
inverse participation ratio already logged here. It was attributed to
`arxiv:2601.19942`, a weak source; the dependency is unnecessary and is removed.

**5.4 The MoE novelty claim was stale.** See §3 — the phase-diagram treatment of
expert ecology now exists.

**5.5 Four experts is the wrong regime.** `load_balance` has a collapse floor of
0.25 at `n_experts=4` and `dead_experts` is a 0-to-4 count. The failure modes
this repo wants to study appear at K3's 16-of-896. T2's "no measurable
difference between routed and dense at matched active FLOPs" is true of a
4-expert toy and should not be carried into K3-shaped questions.

**5.6 `CRITICALITY.md` and `SPARSITY.md` disagreed.** §1 of the former claimed
sparsifying attention and operating at criticality are the same operation;
§1 of the latter correctly says concentration and topology are different
objects. `SPARSITY.md` is right; the former is amended.

**5.7 Invalid JSON.** `t5_results.json` carried bare `NaN` (correct value for the
softmax arm's `sparsity`, invalid JSON, rejected by `jq`). `t5_sparsity.py` now
maps non-finite floats to `null` on write; a run already in flight keeps the old
writer until it is restarted.

---

## 6. Experiments, cheapest first, with registered predictions

**T2-redux — QK-norm × `β`, paired.** *Run: 4 arms × 4 seeds, verdict in
`suite/RESULTS.md` — P1 holds, P2 fails, P3 splits, P4 noise.* Fixes §5.1. Four arms: {QK-norm, none} ×
{fixed `β`, learned `β`}, paired at each seed as in T3.
*Prediction:* with QK-norm, learned `β` moves substantially further from 1.0×
than the 0.934× measured without it, because the compensating degree of freedom
is gone. *The interesting outcome either way:* if the model still lands at
`participation_frac ≈ 0.20` with that path removed, the attractor reading in
`RESULTS.md` graduates from hypothesis to result; if it cannot, `β` was
load-bearing and T2's null was an artefact. Prerequisite for every other
criticality claim in this repo.

**T6 — sinks as an order parameter, across the existing T5 arms.** Stats only,
no new runs: per-layer maximum attention mass on any single key position,
residual-stream kurtosis, and max activation magnitude.
*Prediction (from §4.2):* sinks form in the `softmax` arm and are absent in the
`sigmoid` arm, which has no simplex to concentrate. If sinks appear in the
sigmoid arm too, the "softmax normalisation causes sinks" account is wrong and
`arxiv:2603.05498`'s architectural-artefact reading is favoured.

**T7 — a KDA layer in `suite/hybrid.py`, instrumented.** Log `ρ(A_t)`, `‖A_t‖`,
their gap, and a Fisher memory curve on a held-out sequence. Arms: K3's
lower-bounded `α`, VLA's unit-spectral-norm construction (`arxiv:2605.11196`), a
learned band, and `Diag(α)`-only as the normal-operator control.
*Prediction:* the `‖A‖ − ρ(A)` gap predicts associative-recall accuracy better
than `ρ(A)` alone, and the `Diag(α)`-only control has a visibly shorter memory
curve at matched `ρ`. This is the one experiment here whose target quantity is a
property of the recurrence rather than of scale, so the suite's size is not
automatically disqualifying.

---

## 7. Scope limit

Four layers, dim 192, seq 128, 500 steps, 4 experts. Nothing measured here
transfers as a number to 93 layers, 1M context and 896 experts. What this suite
can test is *mechanisms* — whether `β` is load-bearing once its redundancy is
removed, whether sinks are the homeostat, whether non-normality buys memory.
Those transfer as hypotheses. The numbers do not.
