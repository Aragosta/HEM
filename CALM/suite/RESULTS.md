# Results

## Reproduction gate — WikiText-2, byte level, tier 1

The gate asks one thing before anything else runs: **does HELM's hyperbolic
advantage appear in our hands at all?** If it does not, no CALM-column
difference can be attributed to geometry, and the suite refuses to report an
interaction rather than producing a number that looks like an answer.

Setup: WikiText-2 official splits, byte level, 12,000 steps = 12,288,000 tokens =
1.14 epochs, two seeds, parameters matched to −0.4%.

| cell | top-1 | BPB | brier_1 | **effective rank** | NaN |
| --- | --- | --- | --- | --- | --- |
| `helm_discrete` s0 | 58.66% | 1.9541 | +0.219 | 3.4 | 0 |
| `helm_discrete` s1 | 58.91% | 1.9514 | +0.188 | 4.1 | 0 |
| `euclid_discrete` s0 | 58.89% | 1.9529 | +0.258 | 18.9 | 0 |
| `euclid_discrete` s1 | 58.70% | 1.9767 | +0.211 | 19.5 | 0 |
| *bigram floor* | *32.22%* | *3.3866* | — | — | — |

```
geometry effect: -0.02%  (seed sd 0.18%)
REPRODUCTION GATE FAILED
```

### What is solid

**Both models genuinely learned.** BPB ≈1.95 against a bigram floor of 3.39,
top-1 58.8% against 32.2%, positive brier scores (real signal, not the mode
collapse that made every earlier CALM row negative), zero non-finite steps, seed
spread 0.18%. This is the first point in the project where a cell cleared a
floor set by someone other than us.

**There is no geometry effect here.** −0.02% against a 0.18% seed spread is a
tie. Not a weak effect — an absent one.

### The finding the accuracy column hides

| | top-1 | BPB | effective rank |
| --- | --- | --- | --- |
| HELM | 58.79% | 1.953 | **3.8** |
| Euclidean | 58.80% | 1.965 | **19.2** |

**HELM reaches identical quality on ~5× fewer representational directions.**
That is consistent with what hyperbolic space actually does — embed hierarchy
compactly — and it is the first mechanistic result about the geometry in this
project rather than another repair. It does not convert into better prediction
at this scale, and it costs 2.8× the wall-clock per step (165.7 ms against
59.8 ms at matched parameters).

Whether a 5× more compact representation is worth having is a different question
from whether it predicts better, and this suite was not built to answer it.

### Why this is not evidence against HELM

Two reasons, and the second is a design mistake of mine rather than a limit of
the hardware.

**Scale.** 449,496 parameters on 12M tokens. HELM's published results are at
120M and 1B on far more data. Architecture comparisons at 450K are not known to
predict anything at 120M.

**Byte level is probably the wrong setting to test HELM in.** HELM's thesis is
that *token* embeddings carry hierarchical, power-law structure that hyperbolic
space embeds with low distortion. 256 byte values have essentially no such
structure. Byte level may remove precisely the property the geometry exists to
exploit — which would make this null partly an artefact of the tokenization,
not only of scale.

Bytes were chosen because HELM's Llama-3 tokenizer is not fetchable in this
environment and bits-per-byte is tokenizer-free, which is right for
comparability across models that do not share a vocabulary. It is the wrong
choice for testing *this particular claim*, and that should have been reasoned
about before the run rather than after it.

### What follows

The GPU tier is now required rather than optional, and needs two changes, not
just more compute:

1. **A real tokenizer**, so token-level hierarchy exists to be exploited.
2. **Scale** — 120M+, WikiText-103, the setting HELM's own results come from.

Until then the honest statement is: *at 450K parameters on byte-level
WikiText-2, HELM and a matched Euclidean transformer are indistinguishable in
quality, HELM is 2.8× slower, and HELM's representation is 5× more compact.*
