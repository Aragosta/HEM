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
| `--rope_impl` | `complex` | `complex` is faster in eager; `real` is the one Inductor can fuse |
| `--fuse_experts` | `True` | One GEMM for the SwiGLU gate/up projections instead of two |
| `--grad_checkpoint` | `False` | Recompute block activations in the backward pass |
| `--compile` | `False` | Wrap the model in `torch.compile` |
| `--balance_update` | `True` | Apply the auxiliary-loss-free routing-bias update (dead code upstream) |

## Results

Attention, measured on CPU at the 120M shape (fp32, batch 2, single layer). CPU
has no FlashAttention kernel, so these are *lower* bounds:

| seq len | reference | optimized | speedup | peak allocation |
| --- | --- | --- | --- | --- |
| 1024 | 97.2 ms | 57.0 ms | **1.71×** | 96.2 → 49.5 MiB |
| 2048 | 363.9 ms | 233.1 ms | **1.56×** | 384.4 → 195.1 MiB |

Whole-model speedup on CPU is only ~1.05×: the sync-elimination and
launch-overhead work has nothing to bite on without an asynchronous device. Those
wins are **unmeasured here** — this was developed on a CPU-only machine. Run the
benchmark on a GPU before quoting a number:

```bash
python benchmarks/bench_helm_mice.py --preset 120m --seq-len 2048 --dtype bfloat16
```

## Verify

```bash
python -m pytest tests/ -q       # 32 tests
```

The attention rewrite is **bit-exact** against the published implementation when
matched for softmax precision and rotary path; the MoE is bit-exact and makes
identical routing decisions.

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
