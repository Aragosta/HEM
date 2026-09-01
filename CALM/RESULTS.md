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

## Review of the energy implementation, and why the seeds scattered

Asked for before Stage 2, and it changed the conclusion.

### The implementation is correct

`CALM/experiments/verify_energy.py` compares our `energy_score` against CALM's
`modeling_energy.py::energy_score` with the target noise injected so both draw
identical samples:

| beta | value max diff | gradient max diff |
| --- | --- | --- |
| 0.5 | 0.000e+00 | **NaN** |
| 1.0 | **0.000e+00** | **0.000e+00** |
| 1.5 | 0.000e+00 | 0.000e+00 |

Bit-identical, values and gradients. The formula was never the problem.

The NaN at `beta=0.5` is present in **CALM's implementation too**, and is worth
knowing about: the pairwise term includes the self-distances `‖x_i − x_i‖ = 0`,
and `d/dx ‖x‖^β` is unbounded at 0 for `β < 1`. At `β = 1` PyTorch's subgradient
for the norm at 0 is 0 and it is safe. CALM's default is 1.0, so nothing is
broken in practice — but `β < 1` would silently produce NaN gradients.

A separate sanity check confirms the rule behaves properly: the score falls
monotonically as the predictive distribution is shifted away from the target
(offset 0.0 → 4.81, 0.5 → 6.38, 1.0 → 10.08, 2.0 → 19.85 in loss).

### Three candidate causes, tested

**A. Evaluation noise — rejected.** Majority voting over 8 samples might just be
a noisy estimator of the mode. It is not: accuracy is flat in the pool size.

| seed | acc@8 | acc@32 | acc@128 |
| --- | --- | --- | --- |
| 0 | 53.12% | 56.52% | 57.34% |
| 1 | 18.07% | 17.12% | 20.65% |

The spread is in the model, not the metric.

**B. Manifold drift — real, but not the cause.** HELM's token embedding is a
`ManifoldParameter` living on the hyperboloid; Stage 1 optimized it with plain
AdamW, which walks it off. The violation `|⟨x,x⟩_L + c|` reached **3.92** — the
embedding was nowhere near the manifold it is supposed to inhabit. Adding a
retraction each step drives it to 3.6e-07, and barely moves accuracy
(53.12% → 55.43%, 18.07% → 17.12%). So it was a genuine defect in the experiment
harness, worth fixing, and not the explanation.

(This affected the *harness* only. HELM's own `train.py` uses a Riemannian
optimizer for `ManifoldParameter`s, as upstream intended; the discrete baseline
here was handicapped in exactly the same way, so the Stage 1 comparison stayed
internally fair.)

**C. Learning rate — the cause.** 5000 steps, retraction on, three seeds:

| lr | seed 0 | seed 1 | seed 2 | mean | spread |
| --- | --- | --- | --- | --- | --- |
| 3e-4 | 20.92% | 14.81% | 18.07% | 17.93% | 6.11% |
| **1e-3** | **99.46%** | **98.23%** | **99.59%** | **99.09%** | **1.36%** |
| 3e-3 | 88.86% | 74.18% | 98.64% | 87.23% | 24.46% |

At **1e-3 the spread collapses from 24.5 points to 1.4**, and the mean reaches
**99.09%** — statistically indistinguishable from the Euclidean control's 99.23%
(spread 1.22%). 3e-4 is simply undertrained at this step budget, so this is a
genuine optimum, not "lower is better".

### Revised Stage 1 conclusion

**There is no geometric pathology.** The energy score trains a hyperbolic
backbone to the same accuracy and the same seed-to-seed stability as a Euclidean
one, once it is given a learning rate suited to the objective rather than one
inherited from cross-entropy. The energy score is markedly more lr-sensitive than
cross-entropy — which is the practical lesson, and the reason Stage 1 looked
alarming.

`stage1_energy_head.py` now defaults to `--lr 1e-3`.

---

## Stage 2a — the K>1 patching path works

Stage 1 was K=1 by design. Stage 2 turns on three things at once: patched input,
a sequence shortened by K, and a patch-level latent target. This runs that
plumbing on the toy task, before any of it costs GPU time.

The HELM-specific piece is the **patch embedding**. CALM concatenates K Euclidean
token embeddings and projects them. HELM's embeddings are Lorentz vectors, and
concatenating those lands on no hyperboloid. So here the K *space-like* parts are
concatenated, the time coordinate is recomputed — giving a valid point in a wider
Minkowski space — and a `LorentzLinear` maps it back onto the model's manifold.
Every activation stays on a manifold, which is the property HELM exists to keep.

5000 steps, lr 1e-3, two seeds:

| K | AE recon | positions | seed 0 | seed 1 | mean | energy loss |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 100.00% | 24 | 99.32% | 99.05% | **99.18%** | 16.46 → 4.18 |
| 2 | 100.00% | 12 | 98.01% | 97.59% | **97.80%** | 19.18 → 4.73 |
| 4 | 99.74% | 6 | 89.22% | 87.34% | **88.28%** | 20.79 → 5.15 |

**The path works at every K**, far above the 1.03% chance rate, and the seed
stability from the lr fix holds throughout (spread 0.27 / 0.42 / 1.88 points).
K=1 reproduces Stage 1's 99.09%, confirming the refactored scaffold is
consistent with the original.

Accuracy falls with K, which is expected: at K=4 the model predicts four tokens
ahead with no intermediate context, and that is exactly the trade that buys 4x
fewer autoregressive steps. **One caveat on the size of that fall**: these
sequences are 24 tokens, so K=4 leaves only 6 positions of context in which to
infer the walk's stride. Part of the drop is that truncation rather than
patching itself, and this task cannot separate the two.

### A bug it caught immediately

`TinyAutoencoder.decode` reshaped with a hardcoded `latent.size(0)`. That is
correct for a `(tokens, latent)` input and wrong for the
`(samples, tokens, latent)` block the energy head produces — invisible at K=1,
where no reshape happens, and a crash the moment patching is switched on. Now
shape-agnostic in the leading dimensions.

This is precisely the class of failure Stage 2a exists to surface: ten minutes on
a 33-dimensional CPU model instead of six hours into a GPU run.

## Stage 2b — the K=4 reconstruction shortfall was a budget artifact

Stage 0 reported 86.29% reconstruction at K=4 and attributed it to budget.
Attributing is not testing, and it mattered: CALM's premise is a near-lossless
autoencoder, and at 86% the language model predicts into a latent space that
loses one token in seven.

Sweeping the budget at fixed K=4 on real WikiText:

| steps | accuracy | |
| --- | --- | --- |
| 600 | 86.29% | (Stage 0's number) |
| 1500 | 92.99% | +6.70 |
| 3000 | 95.67% | +2.67 |
| 6000 | **97.55%** | +1.88 |

Monotonic, still climbing at the largest budget, with the diminishing-returns
shape of something approaching an asymptote well above 99%. **Confirmed: a budget
artifact.** The autoencoder remains a real prerequisite — it has to be trained —
but there is no evidence of a quality ceiling that would undermine Stage 2.

## Stage 2c — BrierLM implemented and validated

`experiments/brierlm.py` implements the likelihood-free metric from
`modeling_calm.py::eval_brier`:

```
brier_k = E[ 1{x1_1..k = y_1..k} + 1{x2_1..k = y_1..k} - 1{x1_1..k = x2_1..k} ]
BrierLM = (brier_1 · brier_2 · brier_3 · brier_4)^(1/4)
```

It needs only samples, so it works for a discrete softmax model as well — which
is what makes it the one metric on which discrete HELM and a CALM-HELM can be
compared at all (`ASSESSMENT.md` §3.1).

Validated against analytically known cases. A uniform model over V tokens has
`brier_k = V^-k` and `BrierLM = V^-2.5`:

| | measured | expected |
| --- | --- | --- |
| brier_1 | 0.25099 | 0.25000 |
| brier_2 | 0.06403 | 0.06250 |
| brier_3 | 0.01548 | 0.01562 |
| brier_4 | 0.00441 | 0.00391 |
| BrierLM | 0.03236 | 0.03125 |

A perfect model scores exactly 1.0, and a collapsed one (always the same token)
scores negative — the collision term doing its job.

**A second bug, in the test itself.** The first version of this check used
V = 97, where a 4-gram collision has probability 97^-4 ≈ 1e-8. So `brier_4` was
empirically 0, the product was 0, and `assert got < 10 * expect` passed without
testing anything. A test that passes vacuously is worse than no test; it is now
run at V = 4, where every order is measurable and each is checked individually.

---

## Where this leaves the plan

| stage | status |
| --- | --- |
| 0 — architecture compatible | **confirmed** (75.8M at HELM's vocab) |
| 0 — tokenizer compatible | **confirmed** (128256, all ids valid) |
| 0 — released checkpoint transfers | **open** — network blocked |
| 1 — hyperbolic backbone trains under energy score | **confirmed** |
| 1 — parity with a Euclidean backbone | **confirmed** at lr 1e-3: 99.09% vs 99.23%, spread 1.36% vs 1.22% |
| 1 — instability explained | **yes** — learning rate, not geometry |
| 2a — K>1 patching path runs end to end | **confirmed** (99.18 / 97.80 / 88.28% at K=1/2/4) |
| 2b — autoencoder reaches high reconstruction at K=4 | **confirmed** — 97.55% and climbing; Stage 0's 86% was budget |
| 2c — BrierLM available for both model types | **done**, validated against closed-form cases |
| 2 — K=4 on a real corpus, at scale | not started; needs a GPU |
| 3 — hyperbolic latent | not started; §3.3 of the assessment first, and see `gmvae/` |

Stage 1 passes cleanly, and the caveat it passed with has been resolved: the
27-point seed spread was a learning rate, not a manifold effect. At lr 1e-3 a
hyperbolic backbone under CALM's energy score is indistinguishable from a
Euclidean one on both mean accuracy and stability.

**Stage 2 is now the right next step.** Carry forward three things learned here:

1. Use **lr ~1e-3** for the energy objective, and re-tune it if the backbone size
   changes — cross-entropy's learning rate does not transfer.
2. Keep `ManifoldParameter`s on the manifold. `train.py` already does this with a
   Riemannian optimizer; any new training loop must not quietly use AdamW on
   them, as the Stage 1 harness did.
3. Keep `beta = 1.0`. Below 1 the energy score's self-distance term has an
   unbounded derivative at zero and produces NaN gradients — in CALM's
   implementation as much as ours.

Stage 2's other constraints are unchanged: it needs a GPU, real training data,
and a second evaluation stack, because CALM is likelihood-free and HELM's
benchmark protocol is not (see `ASSESSMENT.md` §3.1).

GM-VAE (`gmvae/`) was evaluated as a candidate hyperbolic latent. It is an
elegant fit — CALM's `(mean, log_std)` output *is* a point on the Gaussian
manifold, so a hyperbolic latent needs no change to the autoencoder's shape — but
it is **not the next thing to do**: its contribution is numerical stability in
hyperbolic VAEs, and the instability we had turned out not to be geometric. See
`gmvae/ASSESSMENT.md`.
