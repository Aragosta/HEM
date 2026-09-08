# Baseline backup — Kimi-K3-Mini, untouched

A verbatim copy of `src/` as vendored from
https://github.com/pablo-reyes8/kimi-k3-pytorch at commit `bbd79cb`, taken
before any V1 work started. See `../UPSTREAM.md`.

Nothing in `V1/` modifies this, and nothing here imports from `V1/`. It exists
so the baseline can be diffed or restored without going through git history:

    diff -r backup_kimi_k3/src src

should stay empty for as long as V1 is built *alongside* Kimi K3 rather than
*into* it, which is the intent — `V1/` imports the Kimi K3 modules and adds the
continuous-autoregressive path on top.
