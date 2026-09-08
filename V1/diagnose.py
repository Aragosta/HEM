"""Why is the energy head's output decoding to noise?

The first benchmark run produced `brier_1 = 0.0083` at a 128-token vocabulary
— exactly chance — while the energy loss did move (11.18 -> 7.92). A loss that
falls while the output stays random has three candidate explanations, and they
call for different fixes:

1. **budget** — the head is converging, just slowly, and needs more steps;
2. **scale** — the head's samples sit in the right *direction* but the wrong
   magnitude, so the energy score improves while the decoder sees nothing it
   recognises;
3. **wiring** — predictions and targets are misaligned, in which case no
   amount of training helps.

This script separates them by measuring, at intervals: the energy loss, the
distance from the predicted latent to the target posterior mean *relative to
the posterior's own scale*, the decode accuracy of the predicted latent, and
the decode accuracy of the target mean (the ceiling). Wiring shows up as a
relative distance that never falls; scale as a distance that falls while
accuracy does not; budget as both improving together.

    python -m V1.diagnose --calm-steps 1500
"""

from __future__ import annotations

import argparse

import torch

from .autoencoder import PatchAutoencoder
from .config import calm_k3_cpu_small_config
from .data import SyntheticCorpus
from .metrics import brier_scores, sample_pairs_calm
from .model import CalmKimiK3
from .train import _optimizer, train_autoencoder


@torch.no_grad()
def probe(model: CalmKimiK3, tokens: torch.Tensor) -> dict:
    """One diagnostic pass on a fixed holdout batch."""
    model.eval()
    output = model(tokens, compute_loss=True)
    predicted = model.energy_head(output.hidden_states)

    target_mean = output.target_mean
    target_std = torch.exp(output.target_log_std)

    distance = (predicted - target_mean).norm(dim=-1)
    target_scale = target_mean.norm(dim=-1)
    posterior_scale = target_std.norm(dim=-1)

    predicted_logits = model.autoencoder.decode(predicted)
    mean_logits = model.autoencoder.decode(target_mean)
    targets = tokens[:, model.patch_size :]

    draw_a, draw_b, brier_targets = sample_pairs_calm(model, tokens, temperature=1.0)
    orders = brier_scores(draw_a, draw_b, brier_targets)

    model.train()
    return {
        "energy_loss": float(output.loss),
        "distance_to_mean": float(distance.mean()),
        "target_norm": float(target_scale.mean()),
        "posterior_noise_norm": float(posterior_scale.mean()),
        "relative_distance": float((distance / target_scale.clamp_min(1e-6)).mean()),
        "accuracy_predicted": float(
            (predicted_logits.argmax(-1) == targets).float().mean()
        ),
        "accuracy_target_mean": float(
            (mean_logits.argmax(-1) == targets).float().mean()
        ),
        "brier_1": float(orders[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-steps", type=int, default=800)
    parser.add_argument("--calm-steps", type=int, default=1500)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()

    torch.manual_seed(options.seed)
    config = calm_k3_cpu_small_config()
    corpus = SyntheticCorpus(vocab_size=config.vocab_size)
    model = CalmKimiK3(config)

    codec = PatchAutoencoder(config.autoencoder)
    report = train_autoencoder(
        codec, corpus, steps=options.ae_steps, batch=32,
        seq_len=options.seq_len, lr=options.lr, log_every=0, seed=options.seed,
    )
    model.load_autoencoder(codec)
    print(
        f"codec ready: holdout reconstruction "
        f"{report.extra['holdout_reconstruction_accuracy']:.4f} "
        f"in {report.seconds:.0f}s\n"
    )

    holdout = corpus.sample(8, options.seq_len, seed=options.seed + 9_000)
    optimizer = _optimizer(model.trainable_parameters(), options.lr)
    model.train()

    header = (
        f"{'step':>6} {'energy':>9} {'|pred-mean|':>12} {'|mean|':>8} "
        f"{'|noise|':>8} {'rel':>7} {'acc_pred':>9} {'acc_ceil':>9} {'brier_1':>8}"
    )
    print(header)
    print("-" * len(header))

    for step, tokens in enumerate(
        corpus.batches(options.calm_steps, options.batch, options.seq_len,
                       seed=options.seed + 1_000)
    ):
        if step % options.probe_every == 0:
            values = probe(model, holdout)
            print(
                f"{step:6d} {values['energy_loss']:9.4f} "
                f"{values['distance_to_mean']:12.4f} {values['target_norm']:8.4f} "
                f"{values['posterior_noise_norm']:8.4f} "
                f"{values['relative_distance']:7.3f} "
                f"{values['accuracy_predicted']:9.4f} "
                f"{values['accuracy_target_mean']:9.4f} "
                f"{values['brier_1']:8.4f}",
                flush=True,
            )
        loss = model(tokens).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
        optimizer.step()

    values = probe(model, holdout)
    print(
        f"{options.calm_steps:6d} {values['energy_loss']:9.4f} "
        f"{values['distance_to_mean']:12.4f} {values['target_norm']:8.4f} "
        f"{values['posterior_noise_norm']:8.4f} "
        f"{values['relative_distance']:7.3f} "
        f"{values['accuracy_predicted']:9.4f} "
        f"{values['accuracy_target_mean']:9.4f} "
        f"{values['brier_1']:8.4f}"
    )


if __name__ == "__main__":
    main()
