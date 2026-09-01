# Did it work?

Short answer: **the implementation works and is faithful to CALM. The model is
not yet competitive — and the gap is specific to the HELM backbone, not to
CALM's objective.** This corrects a conclusion drawn earlier in `RESULTS.md`.

---

## 1. Is the implementation faithful to CALM?

Checked against `upstream/models/modeling_energy.py` and `train/train_energy.sh`,
not from memory.

| | CALM | this implementation |
| --- | --- | --- |
| alignment | `labels[:, patch_size:]` vs inputs `[:, :-1]` patches | same: patch *p* predicts patch *p+1* |
| target | frozen AE encoder → `(mean, log_std)` | same |
| head | MLP on hidden + uniform noise, zero-init final layer | transcribed |
| energy score | `d_x − 2·d_y`, `beta=1.0`, `n_y=100` | **bit-identical**, gradients included |
| samples | `num_samples=8` | 8 |

`labels = input_ids.copy()` in their trainer, with the shift done inside the
model — which is what my `_aligned` reproduces. The energy score is verified
bit-identical (`experiments/verify_energy.py`).

## 2. But the scale is not comparable, and that matters

| | CALM (published) | these experiments |
| --- | --- | --- |
| sequence length | 8192 | 24–32 |
| patches per sequence at K=4 | 2048 | 6–8 |
| hidden size | 1024 | 33 |
| latent size | 128 | 32 |
| training steps | 250,000 | 4,000 |
| autoencoder | 30,000 steps, ~15B tokens | 800 steps, ~290k tokens |
| vocabulary | 128,256 | 64–97 |

Three orders of magnitude smaller on nearly every axis. These runs can establish
that the mechanism *runs correctly*; they cannot establish that HELM-CALM is a
good model, and the K=4 numbers in particular — six patches of context against
CALM's 2048 — are not measuring the same thing CALM measures.

## 3. What actually happened

Two tasks, same code, same budget.

**The easy task** (arithmetic walk, `t+1 = t + stride mod V`):

| | accuracy |
| --- | --- |
| discrete HELM (cross-entropy) | 99.73% |
| CALM + Euclidean backbone | 99.23% |
| CALM + HELM backbone | **99.09%** |

**The harder task** (tree-structured language, 64 tokens, K=1):

| | accuracy |
| --- | --- |
| discrete HELM (cross-entropy) | 98.79% |
| CALM + Euclidean backbone, width 36 | 98.99% |
| CALM + Euclidean backbone, width 33 (matched to HELM) | **98.89%** |
| CALM + HELM backbone | **90.73%** (85.74% in a separate 2-seed run) |

Read together:

* **CALM's objective is not the problem.** A Euclidean backbone under the energy
  score matches ordinary cross-entropy on *both* tasks. The head, the
  autoencoder, the energy score and the sampling-based evaluation all work.
* **The HELM backbone is where the loss appears**, and only once the task gets
  harder: level on the arithmetic walk, 8–13 points behind on the tree language.

### This corrects an earlier claim

`RESULTS.md` concluded from Stage 1 that "there is no geometric pathology" and
that a hyperbolic backbone trains "as well and as stably as a Euclidean one".
That was true *of the arithmetic walk*, and it does not generalise. On a task
with structure, the hyperbolic backbone falls behind a width-matched Euclidean
one under the same objective.

### One confound checked and ruled out

The Euclidean control was originally built at width 36 while HELM runs at
dim 33 — of which one coordinate is the Lorentz time component, so 32 are usable.
That handed the control a ~12% capacity advantage. Re-running at width 33 gives
98.89%, essentially unchanged. **The gap is not a width artefact.**

### What is not established

One to two seeds per cell, and run-to-run variation on the HELM-CALM cell is
about 5 points (85.74% vs 90.73%). The gap of 8–13 points exceeds that, so it
looks real, but it is not nailed down. Causes untested: the learning rate was
tuned on the arithmetic walk and may not transfer; the head consumes the Lorentz
vector directly rather than via `logmap0` (`NEXT.md` §4.1); the manifold
constraint may genuinely limit what the hidden state can carry into a Euclidean
head.

## 4. So: did it work?

| question | answer |
| --- | --- |
| Is the implementation faithful to CALM? | **Yes** — verified against their code, energy score bit-identical |
| Does it run end to end? | **Yes** — 15 integration tests, gradients reach every component |
| Does a hyperbolic backbone train under the energy score? | **Yes** |
| Does it match a Euclidean backbone? | **On an easy task yes; on a harder one, no — 8–13 points behind** |
| Is HELM-CALM a good language model? | **Unknown, and untestable at this scale** |
| Does patching preserve HELM's hierarchy? | **Unknown** — the instrument failed, see `HIERARCHY.md` |

The finding that matters for `NEXT.md`: the risk listed there as "patching may
blur what HELM is for" now has a companion that shows up earlier and is cheaper
to investigate — **the hyperbolic backbone underperforms a Euclidean one under
CALM's objective as soon as the task has structure, before patching is even
involved.** That is measurable at small scale, unlike the hierarchy question, and
should be chased before a GPU is spent.
