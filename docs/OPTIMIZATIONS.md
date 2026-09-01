# What was changed, and why

This repo carries two copies of HELM-MiCE:

| Path | What it is |
| --- | --- |
| `helm/reference/` | The published modules, byte-for-byte. The correctness baseline. |
| `helm/modules/` | The optimized implementation. Same maths, same parameter names, interchangeable checkpoints. |

`tests/` checks the second against the first in float64. Nothing below is an
approximation — every change is an algebraic rewrite, a scheduling change, or a
bug fix, and the tests pin that down.

---

## 1. Attention: the hyperbolic scores are ordinary attention in disguise

HMLA scores two points on the Lorentz manifold by the negative of their squared
Lorentzian distance:

```
s_ij = 2c + 2·⟨q_i, k_j⟩_L        ⟨a,b⟩_L = −a₀b₀ + a₁..d · b₁..d
p    = softmax(s / τ + β)
out  = centroid(p, v) = √c · (p @ v) / ‖p @ v‖_L
```

The reference implements this literally, so it materialises the full
`(B, H, N, N)` score matrix — and then a second copy, because the softmax is
taken with `dtype=torch.float32`. Both are kept alive for the backward pass.

Three observations collapse the whole thing into one `scaled_dot_product_attention`:

* **`2c` and `β` are constants along the softmax axis**, so they cancel exactly.
  They contribute nothing to the output and nothing to any gradient. (`β` is a
  learnable scalar that provably cannot move; it is kept for checkpoint
  compatibility and frozen. `test_attention_bias_has_no_effect` pins this.)
* **Negating the query's time coordinate turns the Minkowski inner product into a
  Euclidean one**: `⟨q,k⟩_L = (−q₀, q₁..d) · (k₀, k₁..d)`. So the scores are a
  plain dot product of `(d+1)`-dimensional vectors.
* **`p @ v` is exactly the attention output** when the *projected* (time
  coordinate included) values are used as `V`. The centroid's renormalisation is
  a cheap elementwise epilogue afterwards.

The temperature is folded into the query, because SDPA's `scale` argument takes a
Python float and `softmax_scale` is a learnable tensor.

The result rides on FlashAttention: no score matrix, no fp32 copy of it, and —
when there is no document mask — `is_causal=True` instead of a materialised
`(N, N)` mask, which lets the kernel skip the entire upper triangle.

**Measured** (CPU, fp32, 120M shape, batch 2, single layer; CPU has no flash
kernel, so this is a *lower* bound):

| seq len | reference | optimized | speedup | peak alloc |
| --- | --- | --- | --- | --- |
| 1024 | 97.2 ms | 57.0 ms | **1.71×** | 96.2 → 49.5 MiB |
| 2048 | 363.9 ms | 233.1 ms | **1.56×** | 384.4 → 195.1 MiB |

Verified **bit-exact** against the reference when matched for softmax precision
and rotary path (`test_hmla_naive_complex_is_bit_exact`), and gradient-matched
to 1e-4 in float64.

## 2. MoE dispatch: one device sync per layer instead of E+1

The routing loop was driven by two things that force the GPU to stall:

```python
counts = torch.bincount(indices.flatten(), ...).tolist()   # sync
for i in range(n_experts):
    idx, top = torch.where(indices == i)                   # sync, per expert
```

`torch.where`/`nonzero` has to read the match count back to the host to size its
output. At 16 layers × 8 experts that is ~144 full pipeline stalls per
micro-batch, each one draining the pipe and serialising against the H2D copy of
the next batch.

This version sorts the routing table once (`argsort` + `bincount`), which makes
each expert's tokens a **contiguous slice** — one gather for the whole layer
instead of one per expert — and pays exactly **one** sync per MoE layer.

The per-expert combination order is preserved exactly: with `topk = 2` the
reference folds each expert into the accumulator with a Lorentzian residual,
which is *not* associative, so the ascending-expert order matters and is kept.

Output is bit-exact; routing decisions are identical (`test_moe_matches_reference`).

## 3. Fused SwiGLU gate/up projection

`w1` and `w3` consume the same input and have the same shape, so they are one
concatenated GEMM. This halves the launch count in the part of the model that
dominates its FLOPs, which matters most exactly when it hurts most: when each
expert holds only a few hundred tokens and the kernels are launch-bound rather
than compute-bound.

Each half is initialised separately, because Xavier's bound depends on `fan_out`
and a single fused draw would give a different distribution than upstream's two.
State-dict hooks translate to and from the `w1`/`w3` layout, so **checkpoints move
in both directions**. Agreement with the reference: 4.4e-16 (float64).

## 4. Rotary embeddings: an honest non-result

The repo now has both a complex and a real-arithmetic rotary implementation.
Measured (CPU, fp32, 4×2048×14×64):

```
eager      complex  1.68 ms    real  28.13 ms
compiled   complex  1.23 ms    real   1.59 ms
```

**The complex path wins, so it stays the default.** The real path was written on
the assumption that it would be faster; it is not, in eager, because of the
strided `x[..., 0::2]` views it needs. It is kept and worth selecting because
TorchInductor has no lowering for complex operators (it warns and falls back), so
it is the only path that can fuse into surrounding kernels under
`torch.compile`. Both produce bit-identical output.

## 5. `model.to(dtype)` silently corrupted the model

The reference stores its rotary table as a complex buffer. Any
`.half()`, `.bfloat16()` or `.to(dtype)` call casts it to a real dtype, **discards
the imaginary part**, and leaves rotary embeddings degraded to a cosine rescale —
with nothing but a `UserWarning` on stderr. Accelerate's mixed precision does not
call `.to(dtype)` on the module, so the released training script escapes it, but
any evaluation or inference harness that casts the model does not.

`HelmMiCE._apply` restores the table after any cast, so device moves keep working
and the values are left alone. Regression test:
`test_dtype_cast_preserves_rotary_table`.

## 6. Bugs fixed

| Bug | Effect |
| --- | --- |
| The config defines `mice_inter_dim`; the model reads `moe_inter_dim` | **HELM-MiCE cannot be constructed from its own config** — `AttributeError`, so every `example/train_mice_*.sh` dies on startup |
| `Gate.forward` returns 2 values in eval mode; `LorentzMoE.forward` unpacks 3 | **The published MiCE model raises the moment `.eval()` is called** — it cannot be evaluated or generated from at all |
| `LorentzMLA` rebinds `softmax_scale` from `nn.Parameter` to a plain tensor under YaRN scaling | Temperature silently stops training and vanishes from the state dict; also reads `args.mscale`, which `config/args.py` never defines |
| `LorentzMoE` reads `self.c`, never assigned | `AttributeError` for any `n_shared_experts != 1` |
| `sequence_balance_loss` overwrites its histogram with `indices * (E/(k·N))` | The load-balancing loss computes nothing meaningful, and only broadcasts at all when `topk == n_experts`; `k` is hard-coded to 2 |
| The routing-bias update in `train.py` is commented out | The auxiliary-loss-free half of the balancing strategy never runs; `Gate.update_bias` is dead code |
| `train_util.py` uses `F.pad` without importing `F` | `NameError` in the collator |
| Boolean CLI flags: `int("True")` | **Every script in `example/` fails to start** |
| KV cache commented out | Generation and `lm-evaluation-harness` scoring re-run the whole prefix per token — O(N²) forwards instead of O(N) |

## 7. Training loop

* Loss and routing statistics accumulate **on device** and are read back once per
  optimizer step. The released loop called `.item()` on every micro-batch — 256
  full stalls per step at the default `gradient_accumulation_steps`, plus a
  per-layer histogram moved to the host with `.cpu()`.
* `broadcast_buffers=False`: the only buffers are the rotary table and the causal
  mask, both deterministic functions of the config and identical on every rank.
* `find_unused_parameters` now defaults to `False`. DDP's unused-parameter search
  walks the autograd graph on every backward pass. Freezing the provably-dead
  attention bias removes the one parameter that never received a gradient.
* Optional activation checkpointing (`--grad_checkpoint`) and `torch.compile`
  (`--compile`).

## 8. Import cost

The `hypercore` `__init__` files imported every submodule eagerly, so building a
language model also imported the graph-learning stack (`torch_geometric`,
`torch_scatter`), the vision stack (`torchvision`) and `sklearn`. They now
resolve lazily (`helm/_lazy.py`); the attribute surface is unchanged.

---

## What is *not* faster

* **The MoE still breaks the `torch.compile` graph.** Variable-sized expert
  groups are inherently data-dependent (`bincount` → `tolist` → data-dependent
  branching). Making it static would require a capacity limit and therefore token
  dropping, which changes training dynamics — not a decision to make silently on
  someone else's model. Attention and the dense layers compile fine.
* **Whole-model speedup on CPU is ~1.05×**, because CPU has no flash kernel and
  no asynchronous queue for the sync elimination to matter to. The attention and
  memory wins above are real and measured; the sync-elimination and
  launch-overhead wins are **not measured here** — this machine has no GPU. Run
  `benchmarks/bench_helm_mice.py` on one before quoting a number.
