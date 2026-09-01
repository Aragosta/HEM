# CALM — Continuous Autoregressive Language Models

Working folder for evaluating whether [CALM](https://github.com/shaochenze/calm)
([arXiv:2510.27688](https://arxiv.org/abs/2510.27688), Shao, Li, Meng & Zhou, 2025) can be
applied to HELM-MiCE.

```
upstream/              the CALM reference implementation, vendored (MIT)
gmvae/                 GM-VAE, evaluated as a hyperbolic latent (Stage 3)
estimate_helm_calm.py  measures what CALM's head would do at HELM's shapes
experiments/           Stage 0 and Stage 1, plus the stability diagnosis
ASSESSMENT.md          does it help HELM? — the analysis
RESULTS.md             what Stages 0 and 1 actually found
```

**Status:** Stages 0 and 1 pass — see `RESULTS.md`. CALM's autoencoder is
architecture- and tokenizer-compatible with HELM (75.8M at HELM's vocabulary,
identical 128256-entry Llama-3 tokenizer), and a hyperbolic backbone trains under
CALM's energy score **as well and as stably as a Euclidean one** (99.09% vs
99.23%, seed spread 1.36% vs 1.22%). An initial 27-point seed spread was traced
to the learning rate, not the geometry. Stage 2 is the next step.

## What CALM does

LLMs are bottlenecked by generating one token at a time. CALM predicts a single
**continuous vector representing a chunk of K tokens**, in two stages:

1. A **frozen autoencoder** compresses K tokens into one latent vector (and
   reconstructs them).
2. A **continuous-domain LM** autoregressively predicts those vectors, using a
   small MLP generative head trained with a likelihood-free **energy score**
   instead of a softmax over the vocabulary.

The transformer therefore sees `seq/K` positions and takes `K×` fewer
autoregressive steps. Because there is no likelihood, evaluation uses **BrierLM**
rather than perplexity.

## Why it is worth checking against HELM specifically

HELM-MiCE is unusually exposed to exactly the cost CALM removes. Its `dim=390`
meets a 128256-entry vocabulary, so the LM head is **50M of a 107M model** and
**81% of the forward pass** — see `docs/UPGRADES.md`. Measured at HELM's shape:

| | HELM | CALM head |
| --- | --- | --- |
| head parameters | 50.02 M | **4.04 M** (12× smaller) |
| output activation (2×1024 tok) | 1002 MiB | **8 MiB** (125× smaller) |
| head fwd+bwd | 5142 ms | **1698 ms** (3.03×) |

Reproduce with `python CALM/estimate_helm_calm.py`.

## Attribution

`upstream/` is the CALM reference implementation, MIT licensed — see
`upstream/LICENSE`. `models/` and `train/` are vendored verbatim; the tokenizer,
validation set and figures are not (`upstream/FETCH.md` says how to get them).

```bibtex
@article{shao2025calm,
  title={Continuous Autoregressive Language Models},
  author={Shao, Chenze and Li, Darren and Meng, Fandong and Zhou, Jie},
  journal={arXiv preprint arXiv:2510.27688},
  year={2025}
}
```
