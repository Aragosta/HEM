# Does patching destroy the hierarchy HELM exists to model?

**The question is still open. The instrument built to answer it does not work, and
the control shows why.** This documents what was run, what it did and did not
establish, and what a working version would need — because a negative result
about one's own measurement is worth more than a confident answer from it.

`experiments/test_hierarchy.py` reproduces everything below.

## The design

`NEXT.md` §6 names this as the central research risk: HELM's thesis is that
hyperbolic geometry matches the **token-level** hierarchy of language, and CALM
compresses K tokens into one vector. If the hierarchy lives below the patch
boundary, a HELM-CALM ends up with CALM's efficiency and no reason to be
hyperbolic.

To make that measurable rather than interpretive:

* **A language with a known hierarchy.** Tokens are the leaves of a balanced tree
  (depth 3, branching 4 → 64 tokens), so the true distance between any two tokens
  is known by construction. Sequences are a random walk that mostly stays within
  a subtree and occasionally jumps to a cousin.
* **A knob for where the signal sits.** `locality="within"` makes
  hierarchically-related tokens adjacent, so the dependency lies *inside* a K=4
  patch; `locality="across"` puts them 4 apart, so it spans patch boundaries. If
  patching is what hurts, "within" should suffer far more.
* **Three measurements.** Spearman correlation between tree distance and
  representation distance (hierarchy recovery); Gromov four-point
  δ-hyperbolicity, normalised by diameter (0 = an exact tree); and next-token
  accuracy, so a model that simply failed to learn is not mistaken for one that
  learned without hierarchy.

The apparatus itself is sound: fed the tree's own distance matrix, the
δ-hyperbolicity routine returns **0.0000**, exactly as it must for a tree.

## What came out

4000 steps, lr 1e-3, two seeds, HELM-CALM at each patch size:

| locality | K | AE recon | accuracy | hierarchy recovery | δ |
| --- | --- | --- | --- | --- | --- |
| within | 1 | 100.00% | 85.74% | −0.0076 | 0.1579 |
| within | 4 | 99.17% | **34.88%** | +0.0097 | 0.1404 |
| across | 1 | 100.00% | 81.91% | +0.0001 | 0.1620 |
| across | 4 | 100.00% | **18.42%** | −0.0219 | 0.1426 |

**Hierarchy recovery is zero everywhere** — all four values sit within ±0.022 of
nothing, including at K=1 where no patching happens at all. The comparison the
experiment was built to make cannot be made: there is no hierarchy signal at K=1
for K=4 to degrade.

## Why — the control

The first explanation considered was that the token embedding barely trains, so
the metric was reading its random initialisation. **That was wrong.** Measured
over 300 steps, the embedding moves **10.27%** relative to its own magnitude
while the head goes from exactly zero to nonzero. It trains.

The control that settles it is the one that should have been run first: does
**discrete HELM** — the model whose entire thesis this is, with its ordinary
cross-entropy head and no CALM anywhere — recover the hierarchy on this language?

| discrete HELM, 3000 steps | |
| --- | --- |
| next-token accuracy | **98.79%** (chance 1.56%) |
| hierarchy recovery, at init | +0.0137 |
| hierarchy recovery, trained | **+0.0599** |
| δ-hyperbolicity, at init | 0.1622 |
| δ-hyperbolicity, trained | **0.2190** |

Discrete HELM learns the language essentially perfectly and still shows almost no
hierarchy in its token embedding — and its representation space becomes *less*
tree-like over training, not more.

So the instrument cannot detect hierarchy on this task in this place. Every
HELM-CALM number in the table above is uninformative about the hierarchy
question, in either direction.

## What the run does establish

Three things survive, and they are worth keeping:

1. **Patching is expensive on this task**: 85.74% → 34.88% (within) and 81.91% →
   18.42% (across) going from K=1 to K=4. Both regimes collapse, and "across"
   slightly more than "within" — the opposite of the prediction. Heavily
   confounded, though: 32-token sequences at K=4 leave only 8 positions, so
   context truncation and patching cannot be separated here, the same caveat as
   Stage 2a.
2. **A real gap between objectives on a structured task.** Discrete HELM reaches
   98.79% where HELM-CALM at K=1 reaches 85.74%. On the earlier arithmetic walk
   the two were level (99.73% vs 99.09%). A more structured language separates
   them, which is itself worth knowing before Stage 2.
3. **The measurement code is correct** and reusable — the tree returns δ = 0.

## What a working version needs

Three candidate reasons the signal is absent, none yet distinguished:

* **Wrong place.** Hierarchy may live in the *contextual* representations rather
  than the static embedding table. Measuring hidden states, averaged per token
  over contexts, is the obvious next attempt — and works at K=1. At K=4 there is
  no per-token hidden state, which is precisely the difficulty.
* **Wrong task.** A 64-token vocabulary with a local transition rule may be
  solvable by memorising transitions, with no pressure to represent the tree.
  A language where hierarchy must be *generalised* — held-out subtrees, or
  agreement across long distances — would create that pressure.
* **Wrong scale.** dim=33, 3 layers, 64 tokens. HELM's own claims are at 120M and
  1B parameters on natural language, where hierarchy is a property of the data
  rather than something injected.

A better metric, robust to all three and comparable across K: **the tree distance
between predicted and true token, on errors**. If a model has internalised the
hierarchy, its mistakes should be near-misses within the tree. It is defined
identically at any patch size, needs no representation-space assumption, and
would work on natural language given a reference hierarchy such as WordNet.

## Status

`NEXT.md` §6 risk 1 stands unchanged and unaddressed: **if the hierarchy lives
below the patch boundary, HELM-CALM keeps CALM's efficiency and loses HELM's
reason to exist.** That remains the thing most worth answering before Stage 2,
and it has not been answered here.
