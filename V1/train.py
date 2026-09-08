"""Two-stage training: fit the codec, freeze it, then fit the language model.

The order is not a convenience. The energy score trains the head to match the
codec's posterior, so if the codec were still moving the target would move
with it and the score would be chasing its own tail. CALM trains the
autoencoder first, freezes it, and only then trains the sequence model — and
the codec's reconstruction accuracy is the ceiling on everything after it.

    python -m V1.train --stage both
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch

from .autoencoder import PatchAutoencoder
from .config import CalmK3Config, calm_k3_cpu_tiny_config
from .data import SyntheticCorpus
from .model import CalmKimiK3


@dataclass
class TrainReport:
    """What a training stage produced."""

    steps: int
    final_loss: float
    seconds: float
    extra: dict


def _optimizer(parameters, lr: float, weight_decay: float = 0.01):
    parameters = list(parameters)
    # Decay matrices only. Gains, biases and scalars are excluded by the usual
    # convention; here it also keeps weight decay off the codec's log_std path,
    # where pulling parameters toward zero would silently narrow the posterior.
    decayed = [p for p in parameters if p.ndim >= 2]
    plain = [p for p in parameters if p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def train_autoencoder(
    autoencoder: PatchAutoencoder,
    corpus: SyntheticCorpus,
    steps: int = 400,
    batch: int = 16,
    seq_len: int = 64,
    lr: float = 3e-3,
    log_every: int = 100,
    seed: int = 0,
) -> TrainReport:
    """Stage 1: teach the codec to put `K` tokens in one vector and back."""
    # A codec taken off a CalmKimiK3 arrives frozen, which is how it must be
    # during stage 2 and cannot be during stage 1.
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(True)
    optimizer = _optimizer(autoencoder.parameters(), lr)
    autoencoder.train()
    started = time.time()
    loss_value = float("nan")

    for step, tokens in enumerate(
        corpus.batches(steps, batch, seq_len, seed=seed)
    ):
        output = autoencoder(tokens)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
        optimizer.step()
        loss_value = float(output.loss)
        if log_every and step % log_every == 0:
            print(
                f"  ae step {step:4d}  loss {loss_value:8.4f}  "
                f"recon {float(output.reconstruction_loss):7.4f}  "
                f"kl {float(output.kl_loss):7.4f}  "
                f"acc {float(output.accuracy):.3f}",
                flush=True,
            )

    holdout = corpus.sample(batch, seq_len, seed=seed + 9_000)
    accuracy = autoencoder.reconstruction_accuracy(holdout)
    return TrainReport(
        steps=steps,
        final_loss=loss_value,
        seconds=time.time() - started,
        extra={"holdout_reconstruction_accuracy": accuracy},
    )


def train_calm(
    model: CalmKimiK3,
    corpus: SyntheticCorpus,
    steps: int = 400,
    batch: int = 8,
    seq_len: int = 64,
    lr: float = 3e-3,
    log_every: int = 100,
    seed: int = 0,
) -> TrainReport:
    """Stage 2: train the backbone and the energy head on a frozen codec."""
    optimizer = _optimizer(model.trainable_parameters(), lr)
    model.train()
    started = time.time()
    loss_value = float("nan")

    for step, tokens in enumerate(
        corpus.batches(steps, batch, seq_len, seed=seed)
    ):
        output = model(tokens)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
        optimizer.step()
        loss_value = float(output.loss)
        if log_every and step % log_every == 0:
            print(f"  calm step {step:4d}  energy loss {loss_value:9.4f}", flush=True)

    return TrainReport(
        steps=steps,
        final_loss=loss_value,
        seconds=time.time() - started,
        extra={},
    )


def train_baseline(
    model,
    corpus: SyntheticCorpus,
    steps: int = 400,
    batch: int = 8,
    seq_len: int = 64,
    lr: float = 3e-3,
    log_every: int = 100,
    seed: int = 0,
) -> TrainReport:
    """The same budget spent on a baseline Kimi K3 with a softmax head."""
    optimizer = _optimizer(model.parameters(), lr)
    model.train()
    started = time.time()
    loss_value = float("nan")

    for step, tokens in enumerate(
        corpus.batches(steps, batch, seq_len, seed=seed)
    ):
        logits = model(input_ids=tokens[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            tokens[:, 1:].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss)
        if log_every and step % log_every == 0:
            print(
                f"  baseline step {step:4d}  ce {loss_value:7.4f}  "
                f"ppl {math.exp(min(loss_value, 20)):9.2f}",
                flush=True,
            )

    return TrainReport(
        steps=steps,
        final_loss=loss_value,
        seconds=time.time() - started,
        extra={"perplexity": math.exp(min(loss_value, 20))},
    )


def build(config: CalmK3Config | None = None) -> tuple[CalmKimiK3, SyntheticCorpus]:
    config = config or calm_k3_cpu_tiny_config()
    corpus = SyntheticCorpus(vocab_size=config.vocab_size)
    return CalmKimiK3(config), corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["ae", "calm", "both"], default="both")
    parser.add_argument("--ae-steps", type=int, default=300)
    parser.add_argument("--calm-steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()

    torch.manual_seed(options.seed)
    config = calm_k3_cpu_tiny_config(patch_size=options.patch_size)
    model, corpus = build(config)
    print(
        f"CALM-K3  vocab {config.vocab_size}  d_model {config.d_model}  "
        f"patch {config.patch_size}  latent {config.autoencoder.latent_size}"
    )
    print(
        f"corpus entropy floor {corpus.markov_entropy_floor():.4f} nats "
        f"(perplexity {math.exp(corpus.markov_entropy_floor()):.2f})\n"
    )

    if options.stage in ("ae", "both"):
        print("stage 1 -- patch autoencoder")
        codec = PatchAutoencoder(config.autoencoder)
        report = train_autoencoder(
            codec,
            corpus,
            steps=options.ae_steps,
            batch=max(options.batch, 16),
            seq_len=options.seq_len,
            lr=options.lr,
            seed=options.seed,
        )
        model.load_autoencoder(codec)
        print(
            f"  done in {report.seconds:.1f}s, holdout reconstruction "
            f"accuracy {report.extra['holdout_reconstruction_accuracy']:.4f}\n"
        )

    if options.stage in ("calm", "both"):
        print("stage 2 -- continuous autoregressive model")
        report = train_calm(
            model,
            corpus,
            steps=options.calm_steps,
            batch=options.batch,
            seq_len=options.seq_len,
            lr=options.lr,
            seed=options.seed + 1_000,
        )
        print(f"  done in {report.seconds:.1f}s, final energy loss {report.final_loss:.4f}")


if __name__ == "__main__":
    main()
