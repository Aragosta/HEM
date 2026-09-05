# The baseline: a scaled-down Kimi K3 block

Everything in this folder is a modification of one baseline, so the baseline
has to be stated precisely enough that a reader can tell which parts are Kimi
K3's and which are ours. This file is that statement. The rule followed
throughout: **what is documented is implemented, what is named but not
documented is substituted and flagged, and nothing is invented and presented as
K3's.**

## What Kimi K3 is

Moonshot AI's flagship MoE model, released 16 July 2026, weights 27 July under
the Kimi K3 License as a native MXFP4 checkpoint. 2.8T total parameters, 104B
active per token, 1M context, native vision (MoonViT-V2, 401M).

From the model card:

| | K3 |
| --- | --- |
| layers | 93 (69 KDA + 24 gated MLA, one dense layer) |
| attention hidden | 7168, 96 heads |
| latent MoE dim | 3584 |
| expert hidden | 3072 |
| routed experts | 896, 16 active |
| shared experts | 2 |
| activation | SiTU-GLU |
| vocabulary | 160K |

Two architectural claims are made for it: **Kimi Delta Attention** (hybrid
linear attention, for the length axis) and **Attention Residuals** (for the
depth axis), together with a "Stable LatentMoE framework" and a reported ~2.5x
scaling-efficiency gain over K2.

## What was ported faithfully

**Attention Residuals.** The one component with a full public technical report
(*Attention Residuals*, Kimi Team, arXiv:2603.15031), so it is implemented from
the equations rather than from a description:

```
h_l      = sum_{i<l} alpha_{i->l} v_i
alpha    = softmax_i( w_l . RMSNorm(v_i) )
v_0      = h_1 (token embedding),   v_i = f_i(h_i)  for i >= 1
```

with `f_i` one *sublayer* (attention or MLP, counted separately), `w_l` a
learned per-sublayer pseudo-query in R^d, and RMSNorm applied to the keys so a
large-magnitude source cannot dominate the softmax. **Pseudo-queries are
zero-initialised**, which the report states is required — a uniform average
over sources at step 0 — and which we followed. `AttnResStream` in `model.py`
is this, and the standard-residual control (`attn_res="none"`) is the same
code path with the aggregation replaced by a running sum.

Full AttnRes is implemented, not Block AttnRes. Block AttnRes exists to keep
`O(Ld)` cross-stage communication affordable under pipeline parallelism at
93 layers; at 4–20 sublayers on one CPU there is no such cost, and the report
measures Full AttnRes as the *better* of the two (1.737 vs ~1.746 val loss).
Implementing the compressed variant would have imported its cost without its
reason.

**LatentMoE geometry.** Routed experts live in a down-projected space:
`down: d_model -> d_latent`, SwiGLU experts inside it, `up: d_latent ->
d_model`, plus shared experts on every token. The ratios are K3's, scaled:
`d_latent/d_model = 0.5` (3584/7168) and `d_ff/d_latent = 0.86` (3072/3584).

**Sparsity, scaled not copied.** K3 activates 16 of 896 (1.8%) plus 2 shared.
At `d_model = 96` a 896-expert layer would have ~8 parameters per expert
neuron and nothing to route; we use 16 routed / 2 active / 1 shared (12.5%).
This is a real deviation and it matters for one experiment only (E3, expert
differentiation across loops), where it is the conservative direction: fewer
experts make differentiation *easier* to detect, so a null there is stronger
than a null at K3's sparsity, and a positive result is weaker.

## What was substituted, and why

| K3 | here | why |
| --- | --- | --- |
| 69 KDA + 24 gated MLA | multi-head attention throughout | asked for the MHA configuration; and KDA is a claim about 1M context, untestable at seq 36–128. A linear-attention layer at this scale would be decoration. |
| SiTU-GLU | SwiGLU | "SiTU" is named in the model card without a definition we could reach. Substituting a documented activation is honest; guessing at an undocumented one is not. |
| "Stable LatentMoE" | sigmoid router + aux-loss-free bias load balancing (DeepSeek-V3 / K2 lineage) | the *name* is public, the mechanism is not. What is implemented is the standard mechanism of that lineage, labelled as an inference rather than as K3's. |
| MXFP4 weights / MXFP8 activations | float32 | quantisation-aware training is a deployment claim; float32 removes it as a confound. |
| MoonViT-V2 vision encoder | none | irrelevant to depth recurrence. |
| 160K vocabulary | 256 (bytes) or 38 (synthetic) | forced by scale; see `DESIGN.md` on what this costs. |

## What is ours, not K3's

Everything the study tests, and it is all switchable from `Config`:

- prelude / shared core / coda with input re-injection and random state
  init — **Huginn** (arXiv:2502.05171), not K3;
- sandwich normalisation — **Ouro** (arXiv:2510.25741) reports it as
  load-bearing for recurrent stability, and it is on in every arm here;
- `step_routing` (loop-index-conditioned MoE routing), `registers`,
  `loop_memory`, and both halting gates — proposals from the brief this folder
  was asked to test. None of them is in K3.

## The one place K3 and the brief meet

The brief's §7.3 argues that a looped model has "nowhere to write": depth
without space. AttnRes is, structurally, an answer to a neighbouring problem —
it gives every layer content-addressed *read* access to every earlier layer's
output instead of one summed state. Run inside a loop, that becomes read access
across loop iterations, which is why the baseline was worth taking from K3
specifically rather than from any MoE transformer. E2 tests exactly this, and
it is the one experiment here whose question would not exist without the
baseline choice.

## Sources

- Kimi K3 model card and config — https://github.com/MoonshotAI/Kimi-K3
- *Attention Residuals*, Kimi Team, arXiv:2603.15031 —
  https://arxiv.org/abs/2603.15031 (equations 2–6, zero-init requirement,
  RMSNorm-on-keys ablation, Full vs Block comparison)
- Kimi K3 release summary (2.8T/104B, KDA + AttnRes, 16-of-896 experts) —
  https://www.marktechpost.com/2026/07/16/moonshot-ai-releases-kimi-k3-a-2-8-trillion-parameter-open-moe-model-with-kimi-delta-attention-and-1m-context/
- Kimi Linear (the architecture AttnRes was integrated into), arXiv:2510.26692
