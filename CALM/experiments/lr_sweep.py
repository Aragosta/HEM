#!/usr/bin/env python3
"""Does a lower learning rate collapse the CALM-on-HELM seed spread?

`diagnose_stability.py` narrowed the Stage 1 instability to the learning rate:
on the seed that stalled, accuracy went 39.4% -> 18.2% -> 1.2% as lr went
1e-3 -> 3e-3 -> 1e-2. 3e-3 was inherited from the discrete model, where
cross-entropy tolerates it fine.

This sweeps lr across seeds, with the embedding kept on the manifold by
retraction, to see whether the spread is explained.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_stability import accuracy, train_calm  # noqa: E402
from stage1_energy_head import make_batches, pretrain_autoencoder  # noqa: E402
from tests._config import tiny_args  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    cli = parser.parse_args()

    args = tiny_args()
    batches = make_batches(args.vocab_size, n_batches=16)
    autoencoder, ae_accuracy = pretrain_autoencoder(args.vocab_size, batches)
    print(f"frozen K=1 autoencoder: reconstruction {ae_accuracy:.2%}")
    print(f"{cli.steps} steps, embedding kept on the manifold by retraction\n")

    header = f"{'lr':>8s} " + " ".join(f"{'seed ' + str(s):>9s}" for s in cli.seeds)
    print(header + f" {'mean':>9s} {'spread':>8s}")
    print("-" * len(header + "     mean   spread"))
    for lr in cli.lrs:
        accs = []
        for seed in cli.seeds:
            model, head, _ = train_calm(args, batches, autoencoder, cli.steps, lr,
                                        seed, riemannian=True)
            accs.append(accuracy(model, head, autoencoder, batches, args, 32))
        spread = max(accs) - min(accs)
        cells = " ".join(f"{a:8.2%} " for a in accs)
        print(f"{lr:8.0e} {cells}{statistics.mean(accs):8.2%} {spread:7.2%}")


if __name__ == "__main__":
    main()
