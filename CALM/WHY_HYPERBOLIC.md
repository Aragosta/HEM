# Why hyperbolic, really — and where it could help a standard MoE+MHA decoder

Read from primary sources, with the numbers rather than the folklore.

## 1. The biological evidence is real, and stronger than usually stated

**Hippocampus.** Zhang, Rich, Lee & Sharpee, *Nature Neuroscience* 2023: CA1
place-cell representations of space "exhibit a **hyperbolic geometry** that
**expands with experience**". Not an embedding artefact — a property of the
neural code, and it grows with exposure.

**Olfaction.** Zhou, Smith & Sharpee, *Science Advances* 2018: both natural odour
statistics and human perceptual descriptions map to a **3-dimensional hyperbolic
space**.

**The fly connectome — the case you asked about.** Sulyok, Balogh & Palla,
*Network geometry of the Drosophila brain* (arXiv 2602.16417, Feb 2026), on the
FlyWire reconstruction: 139,255 neurons, 5·10⁷ synapses. Embedding quality on
the synaptic graph:

| embedding | MA | greedy routing | GR score | edge-pred AUC | EPR5 |
| --- | --- | --- | --- | --- | --- |
| **real 3D anatomy** | 0.363 | 0.075 | 0.048 | 0.862 | 0.968 |
| Node2vec 2D | 0.364 | 0.030 | 0.023 | 0.853 | 0.898 |
| Node2vec 3D | 0.476 | 0.061 | 0.047 | 0.920 | 0.959 |
| **2D hyperbolic (CLOVE)** | **0.528** | **0.553** | **0.390** | **0.960** | **1.000** |

**A 2-dimensional hyperbolic map of the fly brain describes its wiring better
than the fly's own 3-dimensional anatomy** — and greedy routing succeeds 7×
more often (0.553 vs 0.075). That is a striking result and it is why people are
excited.

## 2. But the same paper contains the deflating number

The authors also swept Euclidean Node2vec across dimensions:

| Node2vec dim | MA | greedy routing | GR score |
| --- | --- | --- | --- |
| 4 | 0.500 | 0.142 | 0.102 |
| 8 | 0.613 | 0.438 | 0.322 |
| **16** | **0.653** | **0.629** | **0.484** |
| 32 | **0.660** | 0.767 | 0.609 |
| 64 | 0.642 | 0.847 | 0.687 |
| 128 | 0.554 | **0.859** | **0.709** |

**Euclidean overtakes 2D hyperbolic somewhere between d=8 and d=16, and peaks
far above it at d=32–128.** The paper says so plainly: the hyperbolic level "is
surpassed somewhere between d=8 and d=16".

So the honest statement of the hyperbolic advantage is not *"hyperbolic
represents hierarchy better"*. It is:

> **Hyperbolic geometry buys dimension efficiency. 2D hyperbolic ≈ 8–16D
> Euclidean. Given 32+ dimensions, Euclidean wins outright.**

That reframing is the whole ballgame for language models, and §4 returns to it.

## 3. And the standard theoretical justification is contested

The usual argument is: real networks are scale-free and small-world; such
networks have a hyperbolic latent geometry; topological shortest paths follow
hyperbolic geodesics ("geometric congruence"); therefore hyperbolic space is
the natural home and greedy routing is near-optimal.

Cannistraci & Muscoloni (arXiv 2005.13255) tested that assumption numerically
for the first time and found it **fails**:

> "contrary to current belief, hyperbolic networks do not demonstrate in general
> geometrical congruence and efficient navigability which ... seem to emerge only
> for power-law exponent close to 2."

Statistically, "the hypothesis of congruence should be **always rejected**" across
every parameter combination they tested. High congruence appears only in the
narrow regime γ ≈ 2.

Separately, Thurston's suggestion that **Solv geometry** may fit connectomes as
well as or better than hyperbolic (arXiv 2407.16077) means hyperbolic is not even
uniquely singled out by the biology.

## 4. What this implies for a standard MoE + MHA decoder-only model

Take a typical decoder: residual width d ≈ 2048–8192, per-head dimension
d_head ≈ 64–128, MLA/KV latent ≈ 64–512, MoE router over 8–256 experts.

**The residual stream is the wrong place, and this is quantitative.** The
crossover in §2 sits at d ≈ 8–16. A modern residual stream is **100–1000× past
it**. Whatever hierarchy language has, a 4096-dimensional Euclidean space has
ample room for it; hyperbolic geometry's exponential volume growth is solving a
problem that does not exist at that width, while adding cost, instability, and a
float32 radius ceiling of ~6 nats.

This is not speculation — it is what we measured. HELM applies hyperbolic
geometry *everywhere*, including the residual stream, and in our controlled T0
it lost 3.1× on perplexity (`suite/RESULTS.md`). Our own measurement also
showed HELM's `LorentzLinear` reduces to a Euclidean `nn.Linear` on the space
part with a derived time coordinate — the geometry is thin precisely where the
dimensions are plentiful.

**The low-dimensional bottlenecks are the right place.** Every one of these sits
in or near the d = 8–128 regime where hyperbolic demonstrably wins:

| component | dimension | why it is a candidate |
| --- | --- | --- |
| **per-head attention space** | 64–128 | each head is independently low-dimensional; heads specialise, and specialisation is hierarchical |
| **MLA / KV latent** | 64–512 | DeepSeek already compresses KV to a latent *because* low dimension is the goal; that is exactly where packing efficiency pays |
| **MoE router** | #experts | expert affinity is a shallow hierarchy over a small set |
| residual stream | 2048–8192 | **no** — far past the crossover |
| token embedding | 2048–8192 | **no** — same reason, though the *vocabulary* hierarchy is real |

**And there is a proven, free computational advantage in exactly this setting.**
Wu et al., *Hyperbolic Neural Population Geometry Benefits Computation*
(arXiv 2606.10238, 2026) show that the Modern Hopfield Network update rule —
which **is** the attention mechanism (Ramsauer et al. 2020) — gains a
**double-exponential capacity term** when defined in hyperbolic space, absent
from all Euclidean associative-memory models. Two reasons: exponential volume
growth, and the Lorentz inner product implicitly carrying a factor ~e^{d(a,b)}.

Critically for engineering:

> "the Lorentz inner product is a linear operation and requires **the same
> computational complexity as the Euclidean inner product**."

And their empirical finding matches §2 exactly:

> "these improvements are **most pronounced when the hidden dimensionality is
> constrained**, suggesting that hyperbolic geometry offers a more efficient
> representation space for information storage **in low dimensions**."

## 5. The concrete proposal

**Do not build a fully hyperbolic LLM. Build a Euclidean MoE+MHA decoder with a
hyperbolic attention *head space* and a hyperbolic KV latent.**

- residual stream, FFN, embeddings, LayerNorm: **Euclidean**, unchanged;
- per-head Q/K space (d_head = 64–128): **Lorentz**, with attention scores from
  the Lorentz inner product — same FLOPs, larger associative-memory capacity;
- MLA latent: **Lorentz**, since a smaller latent at equal fidelity is a direct
  KV-cache saving;
- everything else untouched, so the change is auditable and ablatable.

**Why this is a better bet than HELM's design.** It puts the geometry only where
the dimension count says it can pay, keeps the residual stream in the regime
Euclidean demonstrably wins, avoids the float32 radius ceiling (head spaces are
small and normalised), and has a theorem behind it rather than an analogy.

**The falsifiable prediction**: the advantage should *grow* as d_head shrinks. Sweep
d_head ∈ {16, 32, 64, 128} at fixed total width. If hyperbolic head space helps
at 16 and 32 and vanishes by 128, that is the dimension-efficiency mechanism
confirmed, and it tells you exactly which models it is worth applying to. If it
helps uniformly, the mechanism is something else and the story above is wrong.

That experiment is cheap, it is a modification of a *standard* decoder rather
than a new architecture, and it is the one I would run next.

## 6. Honest summary

Why people are interested: **the biology is genuinely hyperbolic** — hippocampus,
olfaction, and the fly connectome in 2D beating its own 3D anatomy — and
hyperbolic space embeds trees with exponentially less distortion than Euclidean
at the same dimension.

What the enthusiasm usually omits: **the advantage is dimension efficiency and it
disappears by d ≈ 16–32**, the geometric-congruence justification is statistically
rejected outside γ ≈ 2, and Solv geometry competes on the same connectome data.

What that means for LLMs: applying it to a 4096-wide residual stream is applying
it where it cannot help. Applying it to a 64-wide attention head, where there is
a capacity theorem and identical FLOPs, is where it might.
