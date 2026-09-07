# Conclusion

176 runs, three rounds, one baseline. What follows is what survived, what did
not, and what I would tell someone about to build a looped model.

---

## 1. The three findings that survived

**Depth is worth about two loops, and it substitutes for data rather than
compounding with it.** At 2 hops, depth's +0.09 advantage at 1500 steps became
−0.01 at 3000: doubling the data erased it. The best depth never moved right as
the budget grew, at either hop count. This was the brief's central hypothesis
and its own stated kill condition — *"if the curves do not shift with data, the
combination is dead and the answer is that four steps is simply enough"* — and
the curves did not shift. Independently, SMELT (arXiv:2609.01343) loops the
middle half of an MoE **twice** under matched FLOPs, parameters and KV cache at
up to 54B. Two scales, same answer.

**The loop is a contraction, and routing is its shadow.** At R=4, expert-set
overlap rises (0.49 → 0.71 → 0.85) exactly as the relative state update falls
(0.51 → 0.19 → 0.08): r = −0.877 across 24 transitions. Because the router is a
linear map of the state (arXiv:2604.09780), routing convergence and state
convergence are one phenomenon measured twice. The identification survives its
own counterexample — one run *diverged*, and its expert overlap fell in step.

This gives saturation a mechanism: **depth past convergence is a no-op**. Not a
scratchpad running out (E2 looked for that and found nothing), not a capacity
limit — an iteration that has reached its fixed point and keeps re-applying
itself.

**The contraction is governed by attention temperature, and that scalar
dominates everything else here.** A 16× sweep of β moves accuracy from 0.709 to
0.447 against a seed spread of 0.004–0.046 — roughly four times the effect of
any architectural change tested. The optimum is *colder* than the default
1/√head_dim, and gets colder as depth grows (×0.5 at R=1, ×0.25 at R=4): a
weight-shared head run repeatedly over a contracting state must start colder to
stay in the useful regime.

---

## 2. What died

| idea | verdict |
| --- | --- |
| the depth ceiling moves with the token budget (§5 of the brief) | **dead**, and it was the brief's own kill condition |
| depth without space — the loop needs somewhere to write (§7.3) | **no effect found**, on an underpowered test: the two-chain task never left the floor, so this is unmeasured rather than answered |
| a halting rule read off the expert path | **dead** — settle step does not separate right from wrong answers at R=4; the R=2 signal was an artefact of a two-valued scale |
| optimal expert count grows with task complexity (arXiv:2410.13964, at this scale) | **not supported** — best k was 1, 8, 2, 4 across four cells, indistinguishable from noise |
| the expert co-activation graph is modular and structure predicts accuracy | **dead here, for a reason that is ours**: aux-loss-free balancing drove effective experts to 15.29 of 16, so specialisation was suppressed by design |
| Full Attention Residuals at shallow depth | **a pure cost** — 1.7× wall-clock, zero accuracy at eight sublayers |

And one **demotion**: loop-index-conditioned MoE routing was the only round-one
result that repeated (+0.037, then +0.063). At the right temperature it buys
**−0.018**, while the temperature alone buys +0.070. Two routes to one gain, and
the cheaper one is larger. The conclusion "the loop wants a phase distinction
between the first pass and the refinement passes" survives; the conclusion about
*where to put it* does not.

---

## 3. What survived from the papers

- **Ouro's storage/manipulation separation reproduces.** Depth moved composition
  accuracy by up to +0.09 while moving byte-level bits-per-byte by 0.02 against
  a seed spread of 0.03. Depth buys manipulation, not statistics, at 1/5000th of
  Ouro's scale.
- **Ouro's step-indexed halting gate fails to extrapolate, exactly as the brief
  predicted.** Its Q-exit freezes at an identical depth for R=8 and R=12 as for
  R=4 — no logits past the trained index. PonderNet's step-invariant gate keeps
  adapting. This is structural, not a matter of scale.
- **Huginn's zero-shot KL exit is the bar.** On a model trained with no gate at
  all it reaches full-depth accuracy at 2.23 of 4 loops. The trained gates
  matched it rather than beating it, which is why RL on the halting gate (§7.4)
  should wait.
- **Training with any halting objective makes shallow inference work** — 0.51 at
  R=1 against 0.405 ungated. Every loop becomes a legitimate output point.

---

## 4. The revised architecture

The brief's §8 synthesis was: Huginn's backbone, MoE core with step-conditioned
routing, a writable register bank, a step-invariant halting gate, CLP serving,
Ouro's data pipeline. After 176 runs, the evidence supports a much smaller
object:

```
prelude → [shared core] × ~2 → coda        depth 2, not 8
standard residuals                          AttnRes earns nothing under ~30 layers
MoE, balanced                               experts hold facts; they do not specialise here
per-loop attention temperature, cold        the one knob that dominated
zero-shot KL exit                            free, and matched every trained gate
no registers, no cross-loop memory          no effect found (underpowered)
no step-conditioned routing                 subsumed by the temperature
```

The practical rule this implies: **before adding registers, gates, expert
conditioners or depth-wise residuals to a looped model, tune one scalar per head
per loop position, and check whether the thing you were about to build still
buys anything.** In this project it would have saved most of two rounds.

---

## 5. What went wrong, kept on the record

Five errors, four of them mine, each caught by something other than luck:

1. **The task did not train** with one question per context — one supervised
   token in 36. Fixed by asking six questions of one context.
2. **The initialisation was the GPT-2 constant** (`std=0.02`) at `dim=64`, six
   times too small. The model sat at chance while a textbook transformer on
   identical batches hit 99.5%. That is why `probe_reference.py` exists.
3. **The paired arms were not paired.** One RNG stream meant adding a register
   bank shifted every later draw, so "paired differences" were unpaired.
   `tests_recur.py` caught it; E0 was rerun.
4. **T2 was run at a depth that could not answer it** (R=2 gives one
   transition). The R=4 redo flipped two verdicts — T2a from untestable to held,
   T2b from "consistent" to failed.
5. **The report generator duplicated tables instead of replacing them**, because
   generated bodies carry their own headings. Fixed, and the write-up it had
   eaten was restored.

A sixth is not an error but a limit worth as much as any result: **the load
balancer suppressed the specialisation two experiments were trying to measure.**
Sweeping `router_bias_lr` toward zero is the cheapest open experiment in the
folder.

---

## 6. The boundary

Everything above is 0.3M parameters, ~3000 steps, four CPU cores, a 6-entity
in-context composition task and byte-level WikiText-2 at sequence 37–128. A null
here is evidence about this regime and not about 1.4B models — and the
Attention Residuals result is a live demonstration that a depth mechanism can
need depth before it appears at all.

What travels better than the effect sizes is the *shape*: which effects appear
on composition but not on next-byte prediction, which failures are structural
rather than quantitative (a gate indexed by a step number cannot use a step
number it was never given, at any scale), and which interventions turn out to be
the same intervention applied at two points in one pipeline.

What would falsify the central story: a model where β is tuned per loop and the
router bias *still* helps. That would mean the temperature and the routing
conditioner are different mechanisms after all, and the upstream/downstream
reading is wrong. At this scale it does not happen. Whether it happens where
attention has real work to do — long context, many heads, a vocabulary worth
attending over — is the untested part, and the reason the critical scaling in
`../CALM/CRITICALITY.md` is written β ≍ log n rather than as a constant.
