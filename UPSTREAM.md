# Provenance

This repository's contents are the **Kimi-K3-Mini** PyTorch implementation by
[pablo-reyes8](https://github.com/pablo-reyes8), vendored wholesale:

- source: https://github.com/pablo-reyes8/kimi-k3-pytorch
- commit: `bbd79cb8cc964900812fdeb661e84f41d0fe27d9`
- vendored: 2026-09-08
- licence: MIT, retained verbatim in `LICENSE` (Copyright (c) 2026 Kimi-K3 Mini
  contributors). Attribution and the licence text must stay with any derivative.

Everything previously in this repository — the HELM/HEM port, HELM-MiCE, the
CALM work and the CALM research suite — was removed in the same commit. It is
not gone: it remains in this branch's git history, and
`git show <commit>^:<path>` or `git checkout <commit>^ -- <path>` will bring any
of it back. The last commit before the wipe is the one to check out from.

The model this implements is Kimi K3 (`arxiv:2607.24653`): a 3 KDA : 1 Gated MLA
layerwise hybrid, Attention Residuals across depth, and Stable LatentMoE for
sparse channel mixing.
