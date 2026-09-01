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

## 0. Does it compute the same thing?

Yes, and this is worth stating precisely rather than asserting.

**Turn every rewrite off and it is bit-identical.** `attn_impl="naive"`,
`rope_impl="complex"`, `fuse_experts=False`, `fuse_residual=False` selects the
literal published formulation of each piece. What remains is only *scheduling* —
the sorted MoE dispatch, the frozen bias, the mask handling, the non-persistent
buffers — and the output is then bit-for-bit equal to the reference: logits,
routing indices and routing scores all compare with `torch.equal`
(`test_exact_configuration_is_bit_identical`).

**With everything on, the differences are at machine epsilon.** The remaining
optimizations reassociate floating-point arithmetic — a fused GEMM sums in a
different order, SDPA reduces differently, the residual reuses one reduction
instead of taking two. At step 0 the logits are still *bit-identical* and the
gradients differ by ~1e-17 in float64.

**Over a training run, that round-off is amplified — by exactly as much as
round-off is.** Adam normalises by `sqrt(v)`, so a relative 1e-16 difference in a
small-gradient direction becomes an O(1) difference in the update *direction*,
and 60 steps compound it to ~1e-5 in the weights. That is not evidence of a
quality difference; it is the ordinary conditioning of stochastic optimization.
The control makes this concrete — take the reference, move **one weight by a
single ULP** (a relative change of 1.3e-16), and train both:

| | worst weight drift after 60 Adam steps |
| --- | --- |
| reference vs. reference + 1 ULP | 9.5e-06 |
| reference vs. optimized | 6.6e-06 |

The optimized model ends up **closer to the reference than the reference is to
itself** under a one-ULP perturbation, and the loss curves track to 5e-07
throughout (`test_trained_weights_stay_within_round_off`,
`test_loss_curves_match_over_many_steps`). The same order of drift is what you
get from changing BLAS thread count, GPU model, or PyTorch version.

**The one deliberate behavioural difference** is the frozen attention bias. It is
a scalar added to every score before a softmax, so it provably cannot affect any
output or receive any real gradient; upstream trains it on pure round-off, where
it wanders slightly off zero without ever changing a prediction. Freezing it also
removes the one parameter that never receives a gradient, which is what forces
DDP's `find_unused_parameters`. See `test_attention_bias_has_no_effect`.

For **inference on an existing checkpoint**, there is no ambiguity at all: the
state dicts are interchangeable and the forward pass agrees to float32 round-off.

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

Whole model, 126M params, 6 layers, batch 2, same conditions:

| | reference | optimized | speedup |
| --- | --- | --- | --- |
| forward, seq 1024 | 1752.7 ms | 1367.1 ms | **1.28×** |
| forward, seq 2048 | 4058.0 ms | 3015.5 ms | **1.35×** |
| **full training step, seq 1024** | **11887 ms / 7045 MiB** | **5697 ms / 3518 MiB** | **2.09× / 2.0× less memory** |

The training step is where the fused head lands, which is why it gains far more
than the forward pass alone.

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

## 3b. Fused Lorentz residual

Profiling a training step puts ~53% of the time in `mm`/`bmm`/`addmm` — the
irreducible compute — and ~28% in hundreds of small `mul`, `pow`, `sum`, `div`,
`cat` and `copy_` calls. That is the cost of the hyperbolic glue: carrying a
point's time coordinate around, discarding it, and deriving it again.

`LResNet` is the worst offender, and it makes seven passes over the activation
where four suffice:

* it computes `sqrt(c) * ave / denom` over **all** `dim` components and then
  throws the time component away, keeping only `[..., 1:]` to rescale;
* it multiplies, divides, then multiplies again, where `scale`, `sqrt(c)` and
  `1/denom` fold into one per-row coefficient;
* it reduces over the feature axis **twice** — once for the Minkowski inner
  product, once for the new time coordinate. The second is redundant: the
  rescaled space part is a scalar multiple of `ave_s`, so
  `|out_s|² = coef² · |ave_s|²` reuses the first reduction.

`helm/modules/lorentz_ops.py::LorentzResidual` is a drop-in replacement with an
identical state dict. **Measured: 7.48 ms → 5.39 ms (1.39×)** at the 120M shape,
seq 2048; agreement with `LResNet` is 7e-15 in float64, gradients included.

## 3c. The language-model head, which turned out to be the biggest win

HELM-MiCE has an unusually lopsided head. The released 120M configuration pairs
`dim = 390` with a 128256-entry Llama-3 vocabulary, so:

* `head` alone is **50M of the model's 107M parameters** (the embedding is
  another 50M; the six transformer layers share the remaining 7M);
* its GEMM outweighs every transformer layer combined — profiling a two-layer
  model put **81% of the forward pass** in the head;
* `self.head(h).float()` materialises `(batch, seq, vocab)` in float32, which at
  the released training shape (batch 4, 2048 tokens) is **3.9 GiB**, held live
  through the backward pass, for a model whose weights are 0.4 GiB.

`helm/modules/fused_ce.py` never builds it. Pass `labels=` to the model and it

* **drops ignored positions before the GEMM.** Sequence packing leaves a real
  fraction of positions carrying `ignore_index`; the reference computes 128256
  logits for each one and discards the row. A pure FLOP saving proportional to
  that fraction.
* **chunks over the remaining tokens**, so the largest live logit block is
  `chunk_size x vocab`.
* **recomputes the logits in the backward pass** rather than storing them —
  one extra GEMM in exchange for the entire activation.

The arithmetic is unchanged: logits are still promoted to float32 before the
softmax, exactly as `.float()` did. The running loss total is kept in float64
(it is a scalar, so the precision is free), which makes the result *exactly*
independent of `chunk_size` rather than merely close.

**Measured** (CPU, fp32, `dim=390`, `vocab=128256`, batch 2 x 1024 tokens, 14%
ignored):

| | fwd + bwd | peak RSS |
| --- | --- | --- |
| `head(h).float()` then `cross_entropy` | 5060 ms | 3740 MiB |
| fused, `chunk_size=256` | **2715 ms (1.86×)** | **1348 MiB (2.8× less)** |

Gradients match the reference to 6e-08; the loss to 5e-07 (float32 summation
order, and the fused version is the more accurate of the two).

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

## Measured and rejected

Two changes that looked worthwhile and were not. Both were implemented and timed
before being dropped, rather than reasoned about:

* **Skipping the embedding's redundant permutes.** `LorentzEmbeddings.forward`
  does `permute(1, 0).contiguous()`, an `index_select`, then `permute(1, 0, 2)`
  back, none of which the `posit_embed=False` path needs. Removing them: 5.30 ms
  → 5.15 ms (1.03×). The `index_select` over a 128256 x 390 table dominates
  completely; the permutes are noise.
* **Fusing the `wq` and `wkv_a` projections.** They consume the same input, so
  they are one concatenated GEMM in the same way `w1`/`w3` are. Measured on the
  GEMM pair alone: 6.08 ms → 5.59 ms (1.09×), which is a fraction of a fraction
  of the block, and it would cost another pair of state-dict translation hooks.
  Not worth the complexity here — though it is a better trade on a GPU, where
  the win is launch overhead rather than arithmetic.

## What is *not* faster

* **The MoE still breaks the `torch.compile` graph.** Variable-sized expert
  groups are inherently data-dependent (`bincount` → `tolist` → data-dependent
  branching). Making it static would require a capacity limit and therefore token
  dropping, which changes training dynamics — not a decision to make silently on
  someone else's model. Attention and the dense layers compile fine.
* **The MoE dispatch's device-sync elimination is unmeasured.** CPU has no
  asynchronous queue for it to matter to. It is a GPU optimization, and this
  machine has no GPU — run `benchmarks/bench_helm_mice.py` on one before quoting
  a number for it.
* **`is_causal=True` buys nothing on CPU.** Skipping the masked half of the
  attention matrix is a FlashAttention property; the CPU fallback still walks it.
  So the whole-model numbers below are, again, lower bounds.
