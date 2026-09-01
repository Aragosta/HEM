# Upgrades beyond the first optimization pass

The first pass (`docs/OPTIMIZATIONS.md`) rewrote attention, the MoE dispatch and
the training loop. This is what came after, in the order it was found. Every
number below is measured on this machine — CPU, float32 — and every claim has a
test behind it. Where something turned out not to help, it says so.

---

## 1. Proving the output does not change

The per-op parity tests show the forward and backward agree at one point. They do
not show that a training *run* stays on the same trajectory, which is the thing
that actually matters.

**Turn every rewrite off and it is bit-identical.** Each optimization is
individually switchable — `attn_impl`, `rope_impl`, `fuse_experts`,
`fuse_residual`. With all of them off, only *scheduling* changes remain (the
sorted MoE dispatch, the frozen bias, the mask handling, the non-persistent
buffers) and logits, routing indices and routing scores all compare with
`torch.equal`.

**With everything on, step-0 logits are still bit-identical** and gradients
differ by ~1e-17 in float64 — the fused GEMM and SDPA simply sum in a different
order.

**Over a run, Adam amplifies that round-off**, because it normalises by
`sqrt(v)`: a relative 1e-16 difference in a small-gradient direction becomes an
O(1) difference in the update *direction*. So the question is not "does it
drift" but "does it drift more than round-off does". The control answers that —
take the reference, move **one weight by a single ULP**, and train both:

| | worst weight drift after 60 Adam steps |
| --- | --- |
| reference vs. reference + 1 ULP | 9.5e-06 |
| reference vs. optimized | **6.6e-06** |

The optimized model ends up closer to the reference than the reference is to
itself under a one-ULP perturbation, and the loss curves track to 5e-07. The
test calibrates its tolerance against that control rather than a guessed
constant (`test_trained_weights_stay_within_round_off`).

---

## 2. The language-model head — the largest single win

HELM-MiCE is lopsided in an unusual way. The released 120M configuration pairs
`dim = 390` with a 128256-entry Llama-3 vocabulary, so:

* `head` alone is **50M of the model's 107M parameters** — the embedding is
  another 50M, and the six transformer layers share the remaining 7M;
* profiling a two-layer model puts **81% of the forward pass** in the head;
* `self.head(h).float()` materialises `(batch, seq, vocab)` in float32:
  **3.9 GiB** at the released training shape, live through the backward pass,
  for a model whose weights are 0.4 GiB.

`helm/modules/fused_ce.py` never builds it. Passing `labels=` to the model

* **drops ignored positions before the GEMM** — sequence packing leaves a real
  fraction of positions at `ignore_index`, and the reference computes 128256
  logits for each one before discarding the row;
* **chunks over the rest**, capping live logits at `chunk_size x vocab`;
* **recomputes logits in the backward pass** instead of storing them — one extra
  GEMM in exchange for the whole activation.

Arithmetic is unchanged: logits are still promoted to float32 before the softmax,
exactly as `.float()` did. The running total is kept in float64 (free, it is a
scalar), which makes the result *exactly* independent of `chunk_size` rather
than merely close.

| dim=390, vocab=128256, 2x1024 tokens, 14% ignored | fwd + bwd | peak RSS |
| --- | --- | --- |
| `head(h).float()` then `cross_entropy` | 5060 ms | 3740 MiB |
| fused, `chunk_size=256` | **2715 ms (1.86×)** | **1348 MiB (2.8× less)** |

Gradients match to 6e-08, loss to 5e-07.

## 3. Fused Lorentz residual

Profiling put ~53% of a training step in `mm`/`bmm`/`addmm` and ~28% in hundreds
of small `mul`, `pow`, `sum`, `div`, `cat` and `copy_` calls — the cost of
carrying a point's time coordinate around, discarding it, and deriving it again.

`LResNet` makes seven passes over the activation where four suffice:

* it computes `sqrt(c) * ave / denom` over **all** `dim` components, then throws
  the time component away and keeps only `[..., 1:]` to rescale;
* it multiplies, divides, then multiplies again, where `scale`, `sqrt(c)` and
  `1/denom` fold into a single per-row coefficient;
* it reduces over the feature axis **twice** — once for the Minkowski inner
  product, once for the new time coordinate. The second is redundant: the
  rescaled space part is a scalar multiple of `ave_s`, so
  `|out_s|² = coef² · |ave_s|²` reuses the first reduction.

`LorentzResidual` is a drop-in replacement with an identical state dict.
**7.48 ms → 5.39 ms (1.39×)**, agreeing to 7e-15 in float64 including gradients.

Whole model, 126M params, 6 layers, batch 2 x 1024 tokens, one training step:

| | reference | optimized |
| --- | --- | --- |
| training step | 11887 ms | **5697 ms (2.09×)** |
| peak RSS | 7045 MiB | **3518 MiB (2.0× less)** |

## 4. Rotary embeddings: the ranking depends on the execution mode

Measured on one block, seq 512:

| | eager | compiled |
| --- | --- | --- |
| complex rope | 37.5 ms | 34.8 ms (1.08×) |
| real rope | 46.8 ms | **30.6 ms (1.53×)** |

The complex kernel wins in eager; TorchInductor cannot lower complex operators
(it warns and falls back), so the real formulation is the only one that fuses
under `torch.compile`. `--rope_impl auto` now picks whichever suits the mode.
This is the first time the real path — written on the assumption it would be
faster, and reported as a non-result in the first pass — has actually paid off.

---

## 5. The KV cache is now *latent*, which is the point of MLA

The first pass restored the cache upstream ships commented out, but restored the
**naive** one: reconstructed per-head keys and values. That throws away exactly
what Multi-head *Latent* Attention exists for.

The MLA cache stores one shared low-rank latent `c_KV` and one rotary key
`k_pe`, and reconstructs every head's K and V on demand. Per token per layer:

| | stored | latent | naive |
| --- | --- | --- | --- |
| 120M | `r=65, rope=17` vs `H=6, qk=50, v=33` | **81** | 498 |
| 1B | `r=257, rope=65` vs `H=14, qk=194, v=129` | **321** | 4522 |

At 2048 tokens, one sequence, bf16:

| | latent | naive |
| --- | --- | --- |
| 120M, 6 layers | **1.9 MiB** | 11.7 MiB (6.1×) |
| 1B, 15 layers | **18.8 MiB** | 265.0 MiB (14.1×) |

`new_kv_caches(..., mode="latent")` is the default; `mode="naive"` spends the
memory to skip the `wkv_b` reconstruction each step, which only pays off at
short context.

Full absorption of `wkv_b` into the query — the trick that makes DeepSeek's MLA
cheap in *compute* as well as memory — does not transfer cleanly here. HELM
scores with the Lorentzian inner product, whose time component
`sqrt(|k_nope|² + |k_pe|² + c)` is a nonlinear function of `k_nope = W_k c_KV`.
Absorbing it would need the quadratic form `c^T (W_k^T W_k) c`, which at these
shapes (`r=65` vs `d_nope=33`) costs *more* than just reconstructing `k_nope`.
The memory saving is real; the compute saving is not available.

### A bug this uncovered

Adding a multi-token prefill path exposed a real defect: the model switched
`is_causal` off whenever a cache was present, so a prefill block attended
**bidirectionally**, and every later layer then cached keys derived from the
wrong hidden states.

It was invisible to every test in the suite, because feeding one token at a time
never produces a multi-token query block, and greedy decoding only ever reads the
last position — which attends to the whole prefix either way. Fixed, with
`test_prefill_then_decode_matches_full_forward` covering both cache modes.

A second real bug: `view_as_complex` requires a storage offset divisible by 2,
and `.contiguous()` does not provide one — a slice can be contiguous and still
start at an odd offset. The rotary half of the key arrives as the second piece of
a `torch.split`, so whether its offset is even depends on `kv_lora_rank`. At the
released 120M shape (`kv_lora_rank = 65`) it is odd, and single-token decoding
raised `Tensor must have a storage_offset divisible by 2`.

---

## 6. The evaluation path, which the port was missing

Cross-checking against the upstream repository turned up one genuine gap. Three
directories were left behind deliberately — `figure/` (images),
`helm/hypercore/data/` (219 MB of graph-learning datasets, unrelated to the LM)
and `lm-evaluation-harness/` (a 47 MB vendored fork). But the fork was not
purely a copy: it carried **HELM-specific code that exists nowhere else**,
namely `lm_eval/models/helm.py` and a second copy of the model under
`helm_module/`.

That second copy has **drifted from the library**. Most notably the eval copy of
`LorentzMoE` contains the eval-mode fix that the library copy lacks:

```python
if self.training:
    weights, indices, scores = self.gate(x)
else:
    weights, indices = self.gate(x)
```

So the authors did fix the bug that makes the published model unusable outside
training — but only in the vendored fork, leaving the library broken. The eval
copy also returns `counts` where the library returns `indices`, and the 1B eval
config disagrees with `example/train_mice_1B.sh` on `qk_nope_head_dim`,
`v_head_dim` and `n_layers` (recorded in
`helm/eval/presets.py::KNOWN_TRAIN_EVAL_MISMATCHES` rather than silently
reconciled — only the authors can say which is authoritative).

`helm/eval/` replaces the fork with a plugin against whatever `lm_eval` is
installed, so there is one copy of the model. It also fixes what made evaluation
slow and incomplete:

* **Requests are batched.** The original scores one continuation per forward pass
  and ignores its own `batch_size` argument. Multiple-choice benchmarks issue one
  request per answer option, so this dominates evaluation cost.
* **Log-probabilities are gathered vectorised**, not in a Python loop calling
  `float()` once per continuation token.
* **`generate_until` works**, using the latent cache. The original raises
  `NotImplementedError`, which excludes every generative task in the harness.
* **`loglikelihood_rolling` works.** The original cannot run at all — it does
  `out_logits, _, _ = self.model(inp)[:, -1]`, unpacking three values from a
  tensor slice, and feeds one token at a time with no cache, so even with the
  unpacking fixed every step would see a one-token context.

Scoring is verified against a transcription of the original algorithm
(`tests/test_eval.py::test_batched_scoring_matches_one_at_a_time`). Batching is
worth only **1.11×** on CPU — it is a GPU optimization, and this is stated rather
than dressed up.

---

## Measured and rejected

Implemented and timed before being dropped, rather than reasoned about:

* **Skipping the embedding's redundant permutes.** `LorentzEmbeddings.forward`
  does `permute(1, 0).contiguous()`, an `index_select`, then `permute(1, 0, 2)`
  back — none of which the `posit_embed=False` path needs. Removing them:
  5.30 → 5.15 ms (**1.03×**). The `index_select` over a 128256 x 390 table
  dominates completely.
* **Fusing the `wq` and `wkv_a` projections.** They share an input, so they are
  one concatenated GEMM in the way `w1`/`w3` are. On the GEMM pair alone:
  6.08 → 5.59 ms (**1.09×**), a fraction of a fraction of the block, at the cost
  of another pair of state-dict translation hooks. A better trade on a GPU, where
  the win would be launch overhead rather than arithmetic.
* **Compiling a block with the complex rope.** 0.95× — *slower*. Inductor's
  complex fallback costs more than it saves. Only worth it with the real rope.

## Still open

1. **A Triton kernel** fusing the Lorentz project-and-normalize epilogue into the
   attention and GEMM outputs. This is the real answer to the ~28% still sitting
   in elementwise glue — but writing GPU kernels that cannot be run here would be
   guessing.
2. **The MoE graph break.** Variable-sized expert groups are inherently
   data-dependent. Making them static needs a capacity limit and therefore token
   dropping, which changes training dynamics — not a decision to take silently on
   someone else's model.
3. **Everything GPU-shaped remains unmeasured**: the MoE device-sync elimination,
   `is_causal` skipping the masked triangle, and eval batching all need a real
   device to pay off. This was developed CPU-only.
