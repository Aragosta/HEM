# What our numbers actually measure — and what a lab would ask for

Written in answer to a direct question: *is the testing honest, does it reflect
text-LLM purposes, do we handle the metrics AI labs use themselves?*

Short answers: **one serious flaw, now fixed; no; and mostly no.** Details below,
because the distinction between "the code is correct" and "the evaluation means
something" is where this project has been weakest.

## 1. The flaw: there was no held-out set

`seed_variance.py` — the script whose table reached `DID_IT_WORK.md` §5 and the
commit message that pushed it — builds sixteen batches and then uses **the same
sixteen batches for training and for evaluation**:

```python
batches = [lang.sample(2, 32, seed=i) for i in range(16)]
...
for s in range(STEPS):
    l = m.loss(batches[s % 16])      # trained on these
...
for t in batches:                    # evaluated on the same ones
    p, tg = m.predict_tokens(t, n_samples=32)
```

That is 1024 tokens, seen roughly 250 times each over 4000 steps, scored on
themselves. **96.15% and 99.09% are training-set accuracies.** They measure how
much of a fixed small corpus each architecture can memorise under the energy
objective. They do not measure generalisation, and "99.09%" is not a quality
number in any sense a language-modelling paper would recognise.

What survives and what does not:

- **Survives:** both arms were measured identically, so the *comparison* is not
  rigged toward either. A 2.94-point difference in memorisation capacity at 4.6
  standard errors is a real difference in something.
- **Does not survive:** the claim that this says anything about which
  architecture is the better language model. Memorisation capacity and
  generalisation routinely disagree, and a 33-dimensional model on a 64-token
  synthetic grammar is exactly the regime where they disagree most.

`hyperbolic_latent_seeds.py` now draws evaluation sequences from the same grammar
under disjoint seeds (1000+i) and reports train and held-out side by side, so the
gap between the two is visible. `seed_variance.py` is left as it was, as the
record of what produced the published table; `DID_IT_WORK.md` §5 is annotated.

## 2. No, this does not reflect text-LLM purposes

The `HierarchicalLanguage` benchmark is a random walk over the leaves of a
balanced tree. It was built for one narrow purpose — to make "did the model learn
a hierarchy" *measurable*, since the true tree distances are known by
construction — and it is good for that. It is not a language model benchmark, and
it was never meant to be one. Concretely, against a real setup:

| | here | a real run |
| --- | --- | --- |
| vocabulary | 64 leaves (+ specials) | 128256 |
| model width | 33 | 2048+ |
| layers | 3 | 24+ |
| sequence length | 32 | 2048–8192 |
| training tokens | ~1024, repeated | 10^9–10^12, mostly once |
| patch size K | **1** | 4 |
| data | synthetic grammar | web text |

The `K = 1` row deserves emphasis: **at K=1 the model is not doing the thing CALM
exists to do.** CALM's entire claim is that predicting one continuous vector per
K tokens buys a K-fold reduction in autoregressive steps. At K=1 there is no
compression, no step reduction, and no efficiency claim — what is being tested is
only whether the energy-score objective and the hyperbolic geometry can coexist.
That is a legitimate question, and it is the one these experiments answer. It is
not the question of whether HELM-CALM is a good language model, and nothing here
should be read as answering that.

## 3. The metrics labs actually use, and where we stand on each

### Perplexity / bits-per-byte on held-out corpora

The primary number in every LM paper, HELM's included. Held-out WikiText-103,
The Pile, C4, or an in-house validation split.

- **Discrete HELM (original and optimized): available and correct.**
  `helm/eval/scoring.py` implements `rolling_logprob`, and the fused
  cross-entropy head returns an exact loss. Nothing blocks a perplexity run
  except a corpus and a GPU.
- **HELM-CALM: structurally impossible.** An implicit sampler has no density.
  This is not a gap in our implementation — it is the defining property of
  CALM's head, and the reason CALM's own paper reports no perplexity.

### lm-eval-harness zero/few-shot suites

HellaSwag, ARC-easy/challenge, PIQA, WinoGrande, OpenBookQA, LAMBADA, MMLU —
the table every model card carries, and the format of the HELM paper's own
benchmark table.

- **Discrete HELM: the plumbing exists and has not been run.**
  `helm/eval/lm_eval_model.py` is a working harness plugin. It has been tested
  for correctness of the interface, **not** run against any actual task. Every
  quality comparison in `docs/UPGRADES.md` is either a numerical-equivalence
  proof against the original implementation (which is strong evidence about the
  optimizations, and is what those claims rest on) or a citation of the paper's
  published numbers (which is not our measurement).
- **HELM-CALM: mostly inapplicable.** Multiple-choice tasks are scored by
  comparing continuation log-likelihoods, which the model does not have.
  Generative tasks with exact-match scoring would work.

### BrierLM

CALM's answer to the likelihood problem: a proper score estimable from samples
alone, so it is computable for both a discrete and a continuous model.
`CALM/experiments/brierlm.py` implements it faithfully and it is self-tested.
**It has been run only on the synthetic grammar**, never on text — so it is
currently a working instrument with no real measurement taken.

### Efficiency

Throughput, tokens/sec, memory, latency. `benchmarks/` measures the HELM
optimizations directly and those numbers (1.71× attention, 2.09× training step,
etc.) are real CPU measurements of real code. The **CALM** efficiency claim — the
K-fold step reduction that is the entire point — is unmeasured and parked in
`PARKED.md`, and cannot be measured at K=1.

## 4. So what would make this honest at the level a lab would want

In dependency order, roughly increasing cost:

1. **Held-out everything.** Done for the hyperbolic-latent experiment; the rest
   of `experiments/` should follow.
2. **Real text, real K.** WikiText-103 with the Llama-3 tokenizer at K=4 — the
   tokenizer is already shared between HELM and CALM, so this is a data-loading
   task, not a modelling one.
3. **Perplexity for discrete HELM, BrierLM for both**, on that corpus. This is
   the first point at which any number here would be comparable to a published
   one.
4. **The 120M preset on a GPU.** Everything above runs at ~1M parameters on CPU,
   which is below the scale at which architecture comparisons are known to be
   predictive of anything.
5. **lm-eval-harness on discrete HELM**, to check the optimized port reproduces
   the paper's table rather than being asserted to.

Steps 1–3 are days of work and would make the CALM claims meaningful. Steps 4–5
need hardware this session does not have.

## 5. What this does not change

The correctness work stands on its own evidence and is not affected by any of
the above:

- the optimized HELM kernels are proven **numerically equivalent** to the
  originals to ULP-calibrated tolerance — that is a comparison against a
  reference implementation, not against a benchmark;
- the ~9 upstream bugs are demonstrated by construction (an eval-mode crash is a
  crash);
- the geometry in `hyperbolic_latent.py` is checked against `geoopt`, against
  numerical integration of the density, and against the propriety property of
  the energy score.

Those are claims about code being right. Section 2 and 3 are about the model
being good, and on that question this repository currently has very little to
say.
