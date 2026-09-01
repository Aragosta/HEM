# Stage 0 and Stage 1 — results

Both stages from `ASSESSMENT.md` §4, run on this machine. CPU only, no GPU, and
Hugging Face / Zenodo / arXiv are blocked by the network policy — which changes
what Stage 0 can prove, stated explicitly below.

Reproduce:

```bash
git clone https://github.com/shaochenze/calm.git /tmp/calm
python CALM/experiments/stage0_autoencoder.py --calm-repo /tmp/calm
python CALM/experiments/stage1_energy_head.py --steps 5000
```

---

## Stage 0 — does CALM's autoencoder fit HELM's tokenization? **Yes**

### 1. Architecture compatibility — confirmed

Building CALM's autoencoder at HELM's vocabulary:

| | |
| --- | --- |
| `vocab_size=128256`, `hidden=512`, `latent=128`, K=4 | |
| total parameters | **75.8 M** (the paper reports 75M) |
| of which embedding | 65.7 M (87%) |

The published size reproduces at HELM's vocabulary, i.e. HELM's vocabulary is the
one CALM was sized for.

### 2. Tokenizer identity — confirmed

| | |
| --- | --- |
| CALM's vendored tokenizer vocab | **128256** |
| HELM's configured `vocab_size` | **128256** |
| 62 WikiText documents tokenized | 289,083 tokens |
| observed id range | [6, 127146] — every id inside HELM's embedding |

This is the claim that removes the biggest prerequisite: the released CALM
autoencoder operates on exactly the token ids HELM emits, so a prototype needs no
15B-token autoencoder pretraining run.

### 3. Reconstruction — mechanism works; K=4 needs real budget

Trained from scratch on real WikiText, on this CPU:

| K | token reconstruction accuracy |
| --- | --- |
| 1 | **99.56%** |
| 2 | **94.78%** |
| 4 | **86.29%** |

**This is a proxy and its limits matter.** The released model is 75M parameters
trained on ~15B tokens and reports >99.9% at K=4. What ran here is far smaller,
on ~290k tokens over a frequency-truncated 4096-entry vocabulary, for 600 steps.
The 86% at K=4 is a statement about a 54-second training run, **not** evidence
against CALM's number. What it does show is the expected shape — compression gets
harder with K — and that the mechanism works end to end at HELM's tokenization.

Validating the released checkpoint needs network access this machine does not
have. That check remains genuinely open.

---

## Stage 1 — does a hyperbolic backbone train under an energy score? **Yes**

The question worth answering before anything else. CALM's likelihood-free
objective has only ever been run on Euclidean transformers; HELM renormalises
every activation onto a Lorentz hyperboloid, and the interaction was unexplored.

Setup: identical HELM-MiCE backbone, identical data (an arithmetic walk),
identical optimizer and step count. Only the head differs. Patch size **K=1** —
no compression, no sequence-length change, so nothing confounds "the objective
works" with "the compression works". There is no efficiency win at K=1 and none
is claimed; this measures trainability only.

Scored by the same metric for both: next-token accuracy, which the CALM model
reaches by decoding its predicted latent through the frozen autoencoder
(majority vote over the 8-sample pool, approximating CALM's temperature-0 mode).

### Result, 5000 steps, three seeds

| seed | discrete HELM | CALM + Euclidean (control) | CALM + HELM |
| --- | --- | --- | --- |
| 0 | 99.73% | 99.46% | 96.20% |
| 1 | 99.73% | 99.73% | **68.61%** |
| 2 | 99.73% | 98.51% | 94.70% |
| **mean** | **99.73%** | **99.23%** | **86.50%** |
| **range** | 0.00 | 1.22 | **27.59** |

Chance is 1.03%.

**A hyperbolic backbone does train under CALM's energy score** — every seed lands
far above chance, and two of three land within ~4 points of the Euclidean
control. The objection that would have killed this direction did not materialise.

**But it is markedly less stable than the Euclidean control.** The control spans
1.2 points across seeds; HELM spans 27.6, with one seed stalling at 68.6%. That
is not noise around a mean, it is a qualitatively different reliability profile,
and it is the most important thing Stage 1 found.

The cause is not established. Three hypotheses, none tested:

* a high-variance sampling objective interacting with the per-layer
  renormalisation onto the hyperboloid, which rescales gradients in a
  way cross-entropy training never exposed;
* the numerical clamps throughout HELM's Lorentz layers (`clamp_min(1e-4)` and
  friends) truncating gradients that the energy score depends on;
* simply a learning rate tuned for the discrete model and inherited unchanged —
  the most mundane explanation and the first one to rule out.

The final energy losses are consistent with the third: HELM's is still higher at
cut-off (5.25 vs 3.75 on seed 0), i.e. under-converged rather than diverged.

### The control was not optional

The first run used 300 steps and produced this:

| | next-token accuracy |
| --- | --- |
| discrete HELM | 98.37% |
| CALM head, HELM backbone | 2.17% |

Read alone, that says hyperbolic geometry breaks the energy score. It does not.
The Euclidean control at the same budget scored **5.71%** — also near chance. The
binding constraint was the step budget: a sampling-based objective simply needs
far more steps than cross-entropy to move, and cross-entropy converged in 300.

Without the control the honest reading of that run would have been a false
negative, and the direction would have been abandoned on it.

### Why one seed would have been misleading

Seed 0 alone (96.20%) reads as "a small gap, probably convergence". Seed 1
(68.61%) reads as "half-broken". Neither is the finding; the *spread* is. A
single-seed run of this experiment would have supported whichever conclusion the
seed happened to favour.

---

## Where this leaves the plan

| stage | status |
| --- | --- |
| 0 — architecture compatible | **confirmed** (75.8M at HELM's vocab) |
| 0 — tokenizer compatible | **confirmed** (128256, all ids valid) |
| 0 — released checkpoint transfers | **open** — network blocked |
| 1 — hyperbolic backbone trains under energy score | **confirmed** |
| 1 — parity with a Euclidean backbone | **not reached** — 86.5% mean vs 99.2%, and 27-point seed spread |
| 2 — K=4 with a Euclidean latent | not started |
| 3 — hyperbolic latent | not started; §3.3 of the assessment first |

Stage 1 passing is the meaningful outcome: the objection that would have killed
this direction cheaply did not materialise. But it passed with a caveat that
should be resolved before Stage 2, not during it — **training stability**, not
final accuracy, is the open risk for a hyperbolic CALM.

The cheap next step is a learning-rate sweep for the CALM+HELM configuration at
this same scale. If the seed spread collapses, it was never a geometry problem.
If it does not, that is a real interaction worth understanding before spending a
GPU on Stage 2.

Stage 2's other constraints are unchanged: it needs a GPU, real training data,
and a second evaluation stack, because CALM is likelihood-free and HELM's
benchmark protocol is not (see `ASSESSMENT.md` §3.1).
