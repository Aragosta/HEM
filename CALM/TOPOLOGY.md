# Attention topology: what the literature already knows, what we measured, and
# what would actually settle it

Written after the two-hour run produced a graph-level result that looked
stronger than it was. Three things prompted it: the finding needed checking
against prior work, the metric behind it (binary reachability) was obviously
incomplete, and a prediction registered against our own best design turned out
to be wrong when measured.

Everything in §1 is a literature claim with its source. §2 is measured here. §3
is a test design and is not a result.

---

## 1. The literature, and where our result sits in it

### 1.1 The framework is established, and named

**Locality Does Not Imply Reachability** ([arXiv:2606.02680](https://arxiv.org/abs/2606.02680))
formalises exactly the object we built by hand. It defines **structural
dependency sets**

```
R_0(t) = {t},    R_{L+1}(t) = R_L(t) ∪ ⋃_{(s,t) ∈ E_{L+1}} R_L(s)
```

— the closure of positions reachable by following attention edges backwards —
and proves (Lemma 1) that if two inputs agree on `R_L(t)`, the hidden state at
`t` is *identical*. Positionwise MLPs, normalisation and residuals cannot add
source positions. That is the rigorous version of "one layer = one hop", and it
is what makes reachability a bound rather than a tendency.

Their headline case is the mirror image of ours. In **fixed-block** causal
attention, positions `p-1` and `p` across a block boundary are adjacent in
sequence distance and **unreachable at every depth** (their Corollary 1). Ours is
the *rewired* case: in a randomised causal graph, near pairs fail because a
detour would need an intermediate position strictly between two adjacent tokens
and none exists. Same theorem, different mask family, and both say the naive
metric intuition — near is easy, far is hard — is wrong under a causal mask.

**What they have that we did not: the bound becomes a task-level number.**
Theorem 2 constructs a `K`-way boundary-copy distribution and proves any
fixed-block causal transformer has top-1 accuracy **at most `1/K`** and expected
cross-entropy **at least `log K`**. That is the single most useful thing in this
literature for us — see §3.1.

They also warn about something our design ignores: coverage is
**phase-conditioned**. For fixed blocks, source `t-d` is reachable from
`t = jb + r` iff `d ≤ r`, so the fraction of positions at which a given distance
is reachable is `(b-d)/b`. A metric averaged over positions, like ours, hides
this. And they show SWA and boundary repair are **coverage-incomparable** —
neither dominates — so a single task cannot rank masks.

### 1.2 Reachability is the wrong metric on its own, and the fix has a name

**What Dense Graph Do You Need for Self-Attention?**
([arXiv:2205.14014](https://arxiv.org/abs/2205.14014)) defines the capacity
measure we were missing:

* **Computational Complexity** `CC(G) = ρ(G) × κ(G)` — mean degree × diameter,
  i.e. cost per layer times layers needed for all-pairs contact.
* **Information Payload** — for a path, `R(P_ab) = ∏_{v ∈ P, v ≠ a} 1/deg(v)`;
  between a pair, the sum over shortest paths; for the graph, the **minimum**
  over the pairs at maximum distance ("the bucket is as tall as its shortest
  plank").
* **NIP(G) = IP(G) / CC(G)**.

The `1/deg(v)` factor is the softmax being a convex combination: a node with
degree `d` gives each source `1/d` of its output at initialisation. Payload
computes as random-walk probability, `I_ab = [(D⁻¹A)^k]ᵀ_ab`.

**And they predict our exact worry.** On the star graph: *"Star graph has the
highest NIP, however, huge amounts of information flow through the global node
can cause a bottleneck of information transfer. This bottleneck reduces the
Information Payload of star graph to 1/(N-1) of the original."* A hub design
looks unbeatable by diameter and can be crippled by capacity. That is precisely
the objection we registered against our own landmark design — and §2 measures
whether it applies.

### 1.3 The same problem, solved twice in a different field

Over-squashing in GNNs is this measurement with a decade of work behind it:

* **curvature** — negatively curved edges bottleneck an exponentially growing
  neighbourhood ([arXiv:2111.14522](https://arxiv.org/abs/2111.14522));
* **effective resistance** — the global signal that local curvature misses
  ([arXiv:2302.06835](https://arxiv.org/abs/2302.06835), and
  [arXiv:2603.11944](https://arxiv.org/abs/2603.11944) which rewires on it
  directly);
* **width, depth and topology** — width mitigates over-squashing but makes the
  network globally more sensitive; depth does not fix it
  ([arXiv:2302.02941](https://arxiv.org/abs/2302.02941));
* **a direct empirical metric** — the decay rate of mutual information between
  node pairs ([arXiv:2508.09265](https://arxiv.org/abs/2508.09265)).

The transformer-side designs are the same constructions under other names:
**Exphormer** ([arXiv:2303.06147](https://arxiv.org/abs/2303.06147)) builds
sparse graph transformers from expanders plus virtual global nodes;
**Socialformer** ([arXiv:2202.10870](https://arxiv.org/abs/2202.10870)) builds
attention patterns from social-network structure, which is the small-world
construction; **Diffuser** ([arXiv:2210.11794](https://arxiv.org/abs/2210.11794))
argues multi-hop diffusion recovers full-attention expressiveness;
**HopFormer** ([arXiv:2602.02268](https://arxiv.org/abs/2602.02268)) makes the
n-hop receptive field the only structural input.

**Nothing here is unclaimed territory.** What our numbers add is a
side-by-side measurement of several mask families under one fixed edge budget,
with the causal asymmetry made explicit — not a new mechanism.

### 1.4 What production systems actually do

Worth stating because it was mis-stated earlier in this project. Kimi K2 is
dense MLA, no sparsity in the token graph. Kimi K3 is a **layerwise hybrid** —
3 Kimi Delta Attention (linear) layers to 1 Gated MLA (global) layer, 69 KDA +
24 MLA across 93 layers, with a global final layer. Gemma 2 interleaves local
and global layers. Mistral popularised sliding windows; StreamingLLM showed
window-only decoding fails without attention **sinks**; LongNet uses dilation;
NSA and DeepSeek-V4-class systems use learned block selection plus compression
plus a local branch.

Two consequences for us. First, **none of these is "window+global"** in the
Longformer sense, and a decoder cannot use Longformer's global tokens as relays
at all (§2.2). Second, the frontier designs are **layer-heterogeneous**, which
our homogeneous mask family cannot express — the reason we added `layered_reach`.

---

## 2. What we measured

All at `seq_len 128`, depth 4, every mask carrying exactly 2040 edges per layer
unless a layered design is named. `suite/graphs.py`; no training anywhere in
this section.

### 2.1 The causal asymmetry (holds)

Reachability at 4 hops, split by separation:

| mask | near 1–4 | window 5–16 | mid 17–48 | far 49–127 |
|---|---|---|---|---|
| p=0 (pure window) | 1.000 | 1.000 | 1.000 | 0.362 |
| p=0.01 | 0.994 | 1.000 | 1.000 | 0.992 |
| p=1 (random) | **0.227** | **0.382** | 0.779 | 0.997 |

The random causal graph fails at *short* range. To reach the token immediately
before you, a direct edge is the only option — a detour would need a position
strictly between two adjacent tokens. Near pairs have exactly one route, far
pairs have many, so band edges are irreplaceable and shortcuts are substitutable.

### 2.2 Hub placement, not the hub idea (a correction)

`window+global` with hubs at positions 0–3 reaches 0.098 in the far band, worst
of everything tested. The same hub count spread through the sequence reaches
**1.000**. A hub at position `h` can only ever carry information about positions
`≤ h`, so hubs at the start shorten paths to tokens that were already reachable
and relay nothing. Longformer/BigBird global tokens attend *bidirectionally*;
under a causal mask that direction does not exist.

### 2.3 Layer heterogeneity wins by 3× (holds)

Total edges across four layers, for full 4-hop coverage:

| design | edges | vs base | reach |
|---|---|---|---|
| homogeneous window w=32 | 14784 | 1.81× | 1.000 |
| 3× window w=16 + 1 **full** | 14376 | 1.76× | 1.000 |
| homogeneous small-world p=0.03 | 8160 | 1.00× | 1.000 |
| **3× window w=3 + 1 hubs/8** | **2718** | **0.33×** | 1.000 |

The winner separates scales across layers: three layers sweep 9 positions back,
one layer reaches landmarks every 8, and the intervals tile the sequence exactly.
A full-attention layer costs 8256 edges on its own — more than the entire
four-layer local budget — so within a pure sparse-attention budget it is the
expensive way to buy reach.

### 2.4 The over-squashing prediction I registered — **wrong, and measured wrong**

The registered prediction was that the 0.33× landmark design routes everything
through 16 landmarks and is therefore the textbook bottleneck, so it should
underperform its reach number while spread-shortcut masks do not.

Implementing Information Payload settles it. `Φ = M_L ⋯ M_1` with each
`M = D⁻¹A` row-stochastic: `Φ[q,k]` is the share of output `q` attributable to
input `k` under uniform (initialisation) attention.

| design | edges | reach | far-band median Φ | far p10 | far Φ=0 | top-16 column share |
|---|---|---|---|---|---|---|
| p=0 window | 8160 | 0.756 | 0 | 0 | 63.8% | 0.375 |
| p=0.01 small-world | 8160 | 0.971 | 6.7e-04 | 2.4e-05 | 7.5% | 0.394 |
| p=0.03 small-world | 8160 | 1.000 | 2.0e-03 | 5.8e-04 | 0% | 0.418 |
| hubs/16 homogeneous | 7800 | 1.000 | 7.0e-03 | 4.0e-03 | 0% | 0.614 |
| window+global (0–3) | 8160 | 0.655 | 0 | 0 | 90.2% | 0.765 |
| **3× w=3 + 1 hubs/8** | **2718** | 1.000 | **9.4e-03** | 4.2e-03 | 0% | **0.470** |
| 3× w=16 + 1 full | 14376 | 1.000 | 1.0e-02 | 8.1e-03 | 0% | 0.666 |
| full attention ×4 | 33024 | 1.000 | 5.4e-03 | 6.5e-04 | 0% | **0.935** |

**The landmark design is not bottlenecked.** Its far-band influence is second
highest of anything measured — above full attention — and its concentration on
the top 16 columns (0.470) is *lower* than full attention (0.935), hubs/16
(0.614) and window+global (0.765).

The reason the star-graph intuition does not transfer: a star funnels `N-1`
sources through **one** node. This design has 16 landmarks and a degree of 4 in
its local layers, and payload goes as `1/deg` per hop, so low degree means each
surviving edge carries a large share. Concentration and bottlenecking are not
the same thing, and the measure that separates them is per-column mass, not
diameter.

**Full attention is not the influence ceiling either** — it has the *highest*
concentration of any design here, because under uniform attention every token
reads the early positions and mass piles up there. That is the graph-level
shadow of the attention-sink phenomenon.

Two internal checks passed, and finding them was the point of writing the second
metric: the zero-influence fraction reproduces the reachability numbers exactly
(63.8% ↔ 0.362, 90.2% ↔ 0.098), and an earlier version disagreed with them
because it multiplied the layer matrices in forward order. `Φ` must be built
last-layer-first. A homogeneous stack hides that bug completely.

### 2.5 What §2 cannot support

* **Uniform attention is the initialisation, not the model.** T2 measured
  attention entropy falling from 1.00 at init to 0.41 trained. Trained `Φ` is
  far more concentrated than anything above, and the ranking may not survive.
  §2.4 describes the architecture's prior.
* **`n = 128`, depth 4.** Full reachability is nearly free here — a 32-wide
  window achieves it. Everything about which design wins is a statement about a
  regime where the problem is easy.
* **No model was trained.** Not one number here says a mask produces a good
  language model.

---

## 3. How to test it properly

Four experiments, ordered by what they settle per CPU-hour. The first is the
one worth doing.

### 3.1 The reachability bound as an exact prediction — **RUN, and it holds**

Result in `suite/RESULTS.md`. The cross-entropy bound held in all three seeds and
was tight to 0.001 nats in one (3.467 against a floor of 3.466), so the
structural analysis in §2 is now checked against trained models. R3/R4 were not
resolvable: outcomes are bimodal (4/18 runs failed, 5/18 solved), and the only
signal surviving the noise is that full attention is the only mask whose worst
seed stays high — a trainability question, not a topology one.

The original design follows.

Adapted from arXiv:2606.02680's Theorem 2. Instead of comparing masks on a task
and arguing about the differences, **construct a task whose answer requires one
specific `(source, target)` pair**, then read off what each mask must score:

* if the pair is unreachable in `R_L(t)`, top-1 accuracy is **≤ 1/K** and
  cross-entropy **≥ log K**, exactly, for any parameters;
* if reachable, the mask *can* solve it, and whether it *does* is the learning
  question.

This is qualitatively better than any accuracy comparison in this suite so far,
because theory makes a **point prediction with no free parameters**, and both
outcomes are informative: a mask that scores above its bound means our
reachability computation is wrong, and a reachable mask stuck at chance is
over-squashing or an optimisation failure, separable by the next test.

Design: `K = 32` classes, needle at controlled separation `d` and phase `r`,
6 masks × 5 seeds, 1000 steps. **≈ 3 CPU-hours.** Registered prediction:
unreachable cells sit at `1/K` within noise; reachable cells beat it; and the
`p=0` window fails only in the far band, matching its 0.362.

### 3.2 Is influence the sufficient statistic? (the general one)

Bin every `(query, needle)` pair by its predicted `log Φ` and plot retrieval
accuracy per bin, **pooling across masks**.

* If one curve fits every mask, `Φ` is the sufficient statistic for topology and
  the whole design problem reduces to maximising it per edge — a strong,
  reusable claim.
* If the curves separate by mask, topology matters in a way `Φ` does not
  capture, and the residual is the interesting object.

Costs nothing beyond §3.1's runs — it is a re-analysis of the same models with
`Φ` computed per pair. **≈ 0 extra.**

### 3.3 Trained influence versus initialisation influence

Recompute `Φ` from the *learned* attention weights and compare to §2.4's uniform
prediction. This is the direct test of §2.5's first caveat, and it decides
whether any of the graph analysis in this document survives contact with
training. **≈ 1 CPU-hour** on §3.1's checkpoints.

Registered prediction: trained `Φ` is far more concentrated, the far-band medians
drop by an order of magnitude, and the *ranking* of masks is preserved — because
a mask cannot create a path that training could concentrate on. If the ranking
inverts, the graph view is decorative.

### 3.4 Which predictor wins: reach, influence, or effective resistance?

Compute all three per pair — binary reach, `Φ`, and effective resistance
(arXiv:2302.06835) — and ask which best predicts per-pair retrieval accuracy from
§3.1. A metrology question with a clean answer, no new training, and it tells us
which quantity to optimise in any future design. **≈ 1 CPU-hour.**

### 3.5 What is deliberately not proposed

* **A leaderboard against a production design.** Our budget axis (edges) is not
  theirs (KV cache and memory bandwidth at 1M context), the regimes differ by
  four orders of magnitude, and linear-attention layers have full reach at `O(n)`
  cost, so our metric cannot even see the trade K3 is making.
* **The length axis, yet.** `n ∈ {128, 512, 2048}` is the obvious follow-up and
  is where these conclusions are most likely to change, but it is worth running
  only after §3.1 establishes that the bound predicts anything at all.

---

## 4. Standing corrections

Recorded here because they are already in this repo's history:

1. **"`window+global` is the standard design and it is dominated."** Both halves
   wrong. In a decoder it is the attention-sink pattern, whose job is softmax
   stability rather than reach, and a causally-aware hub layout is the best mask
   measured here.
2. **"The landmark design is the textbook over-squashing bottleneck."** Measured
   and false at this scale: its far-band influence is second highest of anything
   tested and its concentration is below full attention's.
3. **zlib understates structured masks.** A stride is a handful of bits to
   describe, yet zlib rates strided and dilated masks above a random small-world
   graph, because a row-major bitmap scan cannot see column-wise periodicity.
   The compressibility frontier as drawn is pessimistic exactly where a designer
   would look.
