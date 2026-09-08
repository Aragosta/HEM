"""CALM-K3 against baseline Kimi K3, on the axes where they are comparable.

What is comparable and what is not is the whole difficulty of this benchmark,
so it is stated up front:

**Comparable.** Parameter counts. Backbone positions per token. Training
throughput at a matched token budget. Sequential steps and wall time to
generate a fixed number of tokens. BrierLM, which needs only samples.

**Not comparable.** Training losses — one is a cross-entropy in nats, the
other an energy distance in latent space, and neither bounds the other.
Perplexity, which CALM-K3 does not have at all. Any table printing them side
by side is printing a category error.

Both models are built from the same `KimiBlockConfig` at the same width, get
the same corpus and the same number of optimiser steps. CALM-K3 additionally
needs its codec trained first; that cost is reported separately rather than
hidden, because it is real and the baseline does not pay it.

    python -m V1.benchmark --calm-steps 300 --baseline-steps 300
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import torch

from src import KimiK3
from src.kimi_k3.config import KimiK3Config

from .autoencoder import PatchAutoencoder
from .config import CalmK3Config, calm_k3_cpu_small_config
from .data import SyntheticCorpus
from .generation import generate, generate_baseline
from .metrics import (
    brier_lm,
    brier_scores,
    sample_pairs_baseline,
    sample_pairs_calm,
)
from .model import CalmKimiK3
from .train import train_autoencoder, train_baseline, train_calm


def build_baseline(config: CalmK3Config) -> KimiK3:
    """A vanilla Kimi K3 on the identical backbone config.

    `pad_token_id` is deliberately left unset. The synthetic corpus contains
    no padding, but it does emit token 0 in the first position sometimes, and
    Kimi K3's validation reads a leading pad id as left padding and refuses
    the batch. Declaring no pad id says what is true of this data.
    """
    return KimiK3(
        KimiK3Config(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            backbone=config.backbone,
            pad_token_id=None,
            eos_token_id=config.eos_token_id,
            enable_vision=False,
            enable_mtp=False,
        )
    )


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def time_training_step(
    step_fn, tokens: torch.Tensor, repeats: int = 5, warmup: int = 2
) -> float:
    """Median-free mean ms/step after a warmup, on a fixed batch."""
    for _ in range(warmup):
        step_fn(tokens)
    started = time.perf_counter()
    for _ in range(repeats):
        step_fn(tokens)
    return (time.perf_counter() - started) / repeats * 1000.0


def structural_report(model: CalmKimiK3, baseline: KimiK3, seq_len: int) -> dict:
    patch = model.patch_size
    return {
        "seq_len": seq_len,
        "patch_size": patch,
        "backbone_positions": {
            "calm": seq_len // patch,
            "baseline": seq_len,
            "ratio": patch,
        },
        "parameters": {
            "calm_trainable": sum(p.numel() for p in model.trainable_parameters()),
            "calm_backbone": count_parameters(model.backbone),
            "calm_energy_head": count_parameters(model.energy_head),
            "calm_frozen_codec": count_parameters(model.autoencoder),
            "baseline_total": count_parameters(baseline),
            "baseline_backbone": count_parameters(baseline.backbone),
        },
    }


def speed_report(
    model: CalmKimiK3,
    baseline: KimiK3,
    tokens: torch.Tensor,
    repeats: int = 5,
) -> dict:
    calm_optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=1e-4)
    base_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-4)

    def calm_step(batch: torch.Tensor) -> None:
        loss = model(batch).loss
        calm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        calm_optimizer.step()

    def baseline_step(batch: torch.Tensor) -> None:
        logits = baseline(input_ids=batch[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1)
        )
        base_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        base_optimizer.step()

    model.train()
    baseline.train()
    calm_ms = time_training_step(calm_step, tokens, repeats)
    baseline_ms = time_training_step(baseline_step, tokens, repeats)
    token_count = tokens.numel()
    return {
        "calm_ms_per_step": calm_ms,
        "baseline_ms_per_step": baseline_ms,
        "speedup": baseline_ms / calm_ms if calm_ms else float("nan"),
        "calm_tokens_per_second": token_count / (calm_ms / 1000.0),
        "baseline_tokens_per_second": token_count / (baseline_ms / 1000.0),
    }


def generation_report(
    model: CalmKimiK3,
    baseline: KimiK3,
    prompt: torch.Tensor,
    new_tokens: int,
) -> dict:
    patch = model.patch_size
    started = time.perf_counter()
    calm_out = generate(model, prompt, max_new_patches=new_tokens // patch)
    calm_seconds = time.perf_counter() - started

    started = time.perf_counter()
    base_out = generate_baseline(baseline, prompt, max_new_tokens=new_tokens)
    baseline_seconds = time.perf_counter() - started

    return {
        "new_tokens": new_tokens,
        "calm_sequential_steps": new_tokens // patch,
        "baseline_sequential_steps": new_tokens,
        "calm_seconds": calm_seconds,
        "baseline_seconds": baseline_seconds,
        "speedup": baseline_seconds / calm_seconds if calm_seconds else float("nan"),
        "calm_output_len": int(calm_out.size(1)),
        "baseline_output_len": int(base_out.size(1)),
    }


def quality_report(
    model: CalmKimiK3,
    baseline: KimiK3,
    holdout: torch.Tensor,
    corpus: SyntheticCorpus,
    temperature: float = 1.0,
) -> dict:
    model.eval()
    baseline.eval()

    calm_a, calm_b, calm_targets = sample_pairs_calm(
        model, holdout, temperature=temperature
    )
    base_a, base_b, base_targets = sample_pairs_baseline(
        baseline, holdout, temperature=temperature
    )

    with torch.no_grad():
        logits = baseline(input_ids=holdout[:, :-1]).logits
        cross_entropy = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), holdout[:, 1:].reshape(-1)
        ).item()

    return {
        "brier_lm": {
            "calm": brier_lm(calm_a, calm_b, calm_targets),
            "baseline": brier_lm(base_a, base_b, base_targets),
        },
        # The geometric mean collapses to 0 the moment any single order does,
        # and at a 128-token vocabulary an undertrained model matches no
        # 4-grams at all. The per-order vector is what carries the signal at
        # this scale; brier_1 is the one to read.
        "brier_by_order": {
            "calm": [round(v, 5) for v in brier_scores(
                calm_a, calm_b, calm_targets).tolist()],
            "baseline": [round(v, 5) for v in brier_scores(
                base_a, base_b, base_targets).tolist()],
        },
        "codec_ceiling_accuracy": model.autoencoder.reconstruction_accuracy(holdout),
        "baseline_only": {
            "cross_entropy_nats": cross_entropy,
            "perplexity": math.exp(min(cross_entropy, 20)),
            "corpus_entropy_floor_nats": corpus.markov_entropy_floor(),
        },
        "note": (
            "perplexity exists only for the baseline; CALM-K3 is likelihood-free, "
            "which is why BrierLM is the shared axis"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-steps", type=int, default=600)
    parser.add_argument("--calm-steps", type=int, default=300)
    parser.add_argument("--baseline-steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    options = parser.parse_args()

    torch.manual_seed(options.seed)
    config = calm_k3_cpu_small_config(patch_size=options.patch_size)
    corpus = SyntheticCorpus(vocab_size=config.vocab_size)
    model = CalmKimiK3(config)
    baseline = build_baseline(config)

    print("=" * 72)
    print(
        f"CALM-K3 vs baseline Kimi K3 | d_model {config.d_model} | "
        f"patch {config.patch_size} | latent {config.autoencoder.latent_size} | "
        f"vocab {config.vocab_size}"
    )
    print(
        f"corpus entropy floor {corpus.markov_entropy_floor():.4f} nats "
        f"(perplexity {math.exp(corpus.markov_entropy_floor()):.2f})"
    )
    print("=" * 72)

    print("\n[1/5] stage 1 -- patch autoencoder (a cost the baseline does not pay)")
    codec = PatchAutoencoder(config.autoencoder)
    ae_report = train_autoencoder(
        codec,
        corpus,
        steps=options.ae_steps,
        batch=max(options.batch, 32),
        seq_len=options.seq_len,
        lr=options.lr,
        log_every=max(options.ae_steps // 3, 1),
        seed=options.seed,
    )
    model.load_autoencoder(codec)
    ceiling = ae_report.extra["holdout_reconstruction_accuracy"]
    print(f"  {ae_report.seconds:.1f}s, holdout reconstruction {ceiling:.4f}")

    print("\n[2/5] stage 2 -- training both models on the same corpus")
    calm_report = train_calm(
        model,
        corpus,
        steps=options.calm_steps,
        batch=options.batch,
        seq_len=options.seq_len,
        lr=options.lr,
        log_every=max(options.calm_steps // 3, 1),
        seed=options.seed + 1_000,
    )
    baseline_report = train_baseline(
        baseline,
        corpus,
        steps=options.baseline_steps,
        batch=options.batch,
        seq_len=options.seq_len,
        lr=options.lr,
        log_every=max(options.baseline_steps // 3, 1),
        seed=options.seed + 1_000,
    )

    print("\n[3/5] structure")
    structure = structural_report(model, baseline, options.seq_len)
    print(json.dumps(structure, indent=2))

    print("\n[4/5] speed")
    batch = corpus.sample(options.batch, options.seq_len, seed=options.seed + 7_000)
    speed = speed_report(model, baseline, batch)
    generation = generation_report(
        model,
        baseline,
        corpus.sample(2, options.patch_size * 2, seed=options.seed + 8_000),
        options.new_tokens,
    )
    print(json.dumps({"training": speed, "generation": generation}, indent=2))

    print("\n[5/5] quality")
    holdout = corpus.sample(16, options.seq_len, seed=options.seed + 9_000)
    quality = quality_report(model, baseline, holdout, corpus)
    print(json.dumps(quality, indent=2))

    summary = {
        "config": {
            "d_model": config.d_model,
            "patch_size": config.patch_size,
            "latent_size": config.autoencoder.latent_size,
            "vocab_size": config.vocab_size,
            "seq_len": options.seq_len,
            "batch": options.batch,
            "lr": options.lr,
            "ae_steps": options.ae_steps,
            "calm_steps": options.calm_steps,
            "baseline_steps": options.baseline_steps,
        },
        "training_wall_clock_seconds": {
            "autoencoder": ae_report.seconds,
            "calm": calm_report.seconds,
            "baseline": baseline_report.seconds,
        },
        "final_train_loss_not_comparable": {
            "calm_energy": calm_report.final_loss,
            "baseline_cross_entropy": baseline_report.final_loss,
        },
        "structure": structure,
        "speed": speed,
        "generation": generation,
        "quality": quality,
    }
    if options.out:
        with open(options.out, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote {options.out}")


if __name__ == "__main__":
    main()
