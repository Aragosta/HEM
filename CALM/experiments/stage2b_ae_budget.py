#!/usr/bin/env python3
"""Stage 2b: is the K=4 reconstruction shortfall a training-budget artifact?

Stage 0 trained a CALM-style autoencoder from scratch on real WikiText and got
86.29% token reconstruction at K=4, against the >99.9% the paper reports for its
released 75M model. That was attributed to budget -- 600 steps on ~290k tokens --
but attributing is not testing, and it matters: CALM's premise is a near-lossless
autoencoder. At 86% the language model would be predicting into a latent space
that loses one token in seven, and no amount of backbone quality recovers that.

This sweeps the training budget at fixed K and watches whether accuracy climbs.

Usage: python CALM/experiments/stage2b_ae_budget.py --calm-repo /path/to/calm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage0_autoencoder import check_tokenizer, train_autoencoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calm-repo", type=Path, required=True)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[600, 1500, 3000, 6000])
    parser.add_argument("--hidden", type=int, default=256)
    cli = parser.parse_args()

    _, ids = check_tokenizer(cli.calm_repo, n_docs=400)

    print(f"Reconstruction vs training budget, K={cli.patch}, hidden={cli.hidden}")
    print(f"real WikiText, {len(ids):,} tokens\n")
    print(f"{'steps':>7s} {'accuracy':>9s} {'seconds':>8s}")
    print("-" * 27)
    previous = None
    for steps in cli.budgets:
        accuracy, _, seconds = train_autoencoder(ids, cli.patch, steps=steps,
                                                 hidden=cli.hidden, log_every=10**9)
        arrow = "" if previous is None else f"  ({accuracy - previous:+.2%})"
        print(f"{steps:7d} {accuracy:8.2%} {seconds:8.0f}{arrow}")
        previous = accuracy

    print("\nIf accuracy is still climbing at the largest budget, Stage 0's 86.29%")
    print("was a budget artifact. If it has plateaued well below ~99%, the")
    print("autoencoder is a real prerequisite cost for Stage 2, not a formality.")


if __name__ == "__main__":
    main()
