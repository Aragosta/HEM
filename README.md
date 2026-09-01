# HELM-MiCE, optimized

A port of [HELM](https://github.com/Graph-and-Geometric-Learning/helm)
("HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts",
[arXiv:2505.24722](https://arxiv.org/abs/2505.24722))
with a rewritten HELM-MiCE — hyperbolic multi-head latent attention (HMLA) plus a
mixture of curvature experts (MiCE) — that runs faster and, unlike the original,
can actually be evaluated.

The published implementation is kept byte-for-byte in `helm/reference/` as the
correctness baseline. Every optimization is an algebraic rewrite or a scheduling
change, not an approximation, and `tests/` pins that down in float64 against the
original.

```
helm/modules/     optimized HMLA, MiCE and the model
helm/reference/   the published modules, unmodified
helm/hypercore/   vendored HyperCore layers (graph datasets dropped)
tests/            32 parity and regression tests
benchmarks/       reference vs optimized, timing + memory + op counts
docs/             what changed and why
```

## Install

```bash
pip install torch geoopt                 # core model
pip install -r requirements.txt          # + training stack (accelerate, llmfoundry, ...)
```

The language model needs only `torch` and `geoopt`. The graph and vision parts of
HyperCore pull in `torch_geometric`, `torch_scatter`, `torchvision` and `sklearn`,
but they are imported lazily, so you do not need them to train or run an LM.

## Use

```python
import torch
from helm.hypercore.manifolds import Lorentz
from helm.modules.helm_mice import HelmMiCE
from config.args import parser

args = parser.parse_args([])
model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))

tokens = torch.randint(0, args.vocab_size, (2, 512))
logits, expert_indices, routing_scores = model(tokens)   # train mode

model.eval()
with torch.no_grad():
    logits = model(tokens)                               # eval mode (upstream raises here)
```

Incremental decoding, using the KV cache that upstream ships commented out:

```python
caches = model.new_kv_caches(max_batch_size=2)
for i in range(tokens.size(1)):
    logits = model(tokens[:, i:i + 1], start_pos=i, caches=caches)
```

Checkpoints are interchangeable with the reference in both directions:

```python
model.load_state_dict(torch.load("upstream.pt")["model_state_dict"], strict=False)
```

## Train

```bash
bash example/train_mice_120M.sh
```

or directly, with the optimization flags:

```bash
accelerate launch --mixed_precision bf16 train.py \
    --model_name HELM_MiCE --dim 390 --n_layers 6 --n_heads 6 \
    --attn_impl flash --fuse_experts True --grad_checkpoint False --compile False
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--attn_impl` | `flash` | `flash` fuses the hyperbolic scores into `scaled_dot_product_attention`; `naive` is the literal published formulation |
| `--rope_impl` | `auto` | `complex` in eager, `real` under `torch.compile` — measured, they invert |
| `--fuse_experts` | `True` | One GEMM for the SwiGLU gate/up projections instead of two |
| `--grad_checkpoint` | `False` | Recompute block activations in the backward pass |
| `--compile` | `False` | Wrap the model in `torch.compile` (with the real rope, 1.53× on a block) |
| `--ce_chunk_size` | `512` | Tokens per block in the fused head |
| `--balance_update` | `True` | Apply the auxiliary-loss-free routing-bias update (dead code upstream) |

## Results

Measured on CPU at the 120M shape (fp32, batch 2). CPU has no FlashAttention
kernel, so these are *lower* bounds:

| | reference | optimized | speedup |
| --- | --- | --- | --- |
| attention, seq 1024 | 97.2 ms | 57.0 ms | **1.71×** |
| attention, seq 2048 | 363.9 ms | 233.1 ms | **1.56×** |
| whole model forward, seq 1024 | 1752.7 ms | 1367.1 ms | **1.28×** |
| whole model forward, seq 2048 | 4058.0 ms | 3015.5 ms | **1.35×** |
| **full training step, seq 1024** | **11.9 s / 7045 MiB** | **5.7 s / 3518 MiB** | **2.09× / 2.0× less memory** |

Two things drive the training-step number. Attention never builds its
`(B, H, N, N)` score matrix, so its peak allocation roughly halves (384 → 195 MiB
at seq 2048). And the LM head — 50M of the model's 107M parameters, because
`dim=390` meets a 128256-entry vocabulary — no longer materialises **3.9 GiB** of
float32 logits; `labels=` runs a chunked fused cross-entropy that also skips the
padded positions before the projection instead of after.

Two wins here are **unmeasured**, because this was developed on a CPU-only
machine: eliminating the MoE's per-expert device syncs needs an asynchronous
device to matter, and `is_causal=True` only skips the masked half of the
attention matrix under a real FlashAttention kernel. Run the benchmark on a GPU
before quoting a number:

```bash
python benchmarks/bench_helm_mice.py --preset 120m --seq-len 2048 --dtype bfloat16
```

## Is the output the same?

Yes — and the tests are built to demonstrate it rather than assert it.

```bash
python -m pytest tests/ -q       # 55 tests
```

* **Turn every rewrite off and it is bit-identical.** `attn_impl="naive"`,
  `rope_impl="complex"`, `fuse_experts=False`, `fuse_residual=False` gives the
  literal published formulation; what remains is pure scheduling, and logits,
  routing indices and routing scores all match under `torch.equal`.
* **With everything on, step-0 logits are still bit-identical** and gradients
  differ by ~1e-17 in float64 — the fused GEMM and SDPA just sum in a different
  order.
* **Over a training run that round-off gets amplified, by exactly as much as
  round-off is.** Take the reference, move **one weight by a single ULP**, and
  train both for 60 Adam steps:

  | | worst weight drift |
  | --- | --- |
  | reference vs. reference + 1 ULP | 9.5e-06 |
  | reference vs. optimized | 6.6e-06 |

  The optimized model ends up closer to the reference than the reference is to
  itself under a one-ULP perturbation, and the loss curves track to 5e-07. This
  is the same order of drift you get from changing BLAS threads or GPU model.

The one deliberate difference is the attention bias, a scalar added to every
score before a softmax — it provably cannot change any output, so it is frozen
here while upstream trains it on round-off. Details in
[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md).

## What was wrong with the original

Nine bugs surfaced while porting, including one that makes the released model
unusable outside training (`Gate` returns 2 values in eval mode, `LorentzMoE`
unpacks 3), one that stops it being built from its own config at all
(`mice_inter_dim` vs `moe_inter_dim`), and one that silently corrupts the model
on `.bfloat16()`. The full
list, with what each one breaks, is in
**[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md)**.

## Credit

Model, method and the HyperCore layers are the work of the HELM authors
(Yale Graph and Geometric Learning group), Apache-2.0 — see `LICENSE`. If you use
this, cite the paper:

```bibtex
@article{he2025helm,
  title={HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts},
  author={He, Neil and Anand, Rishabh and Madhu, Hiren and Maatouk, Ali and Krishnaswamy, Smita and Tassiulas, Leandros and Yang, Menglin and Ying, Rex},
  journal={arXiv preprint arXiv:2505.24722},
  year={2025},
}
```
