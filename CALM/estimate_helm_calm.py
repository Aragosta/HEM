#!/usr/bin/env python3
"""What would CALM's continuous head do to HELM-MiCE?

CALM replaces next-*token* prediction over a 128256-entry vocabulary with
next-*patch* prediction of a continuous latent vector, produced by a small MLP
generative head trained with an energy score. This script measures the two
consequences that matter for HELM, using the real modules rather than estimates:

1. **The head.** HELM's ``Linear(dim, vocab)`` is 50M of a 107M model and, when
   profiled, 81% of the forward pass. CALM's head outputs ``latent_size`` (128)
   instead of 128256.
2. **The sequence.** Predicting K tokens per step shortens the transformer's
   sequence by K, which is quadratic in the attention scores and linear
   everywhere else.

Run: ``python CALM/estimate_helm_calm.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.eval.presets import preset_args  # noqa: E402


class CalmHead(nn.Module):
    """CALM's ``MLPGenerator``, transcribed, at an arbitrary hidden size.

    Source: ``CALM/upstream/models/modeling_energy.py``. Reproduced here so the
    parameter count and latency can be measured at HELM's shapes without
    importing the upstream HF stack.
    """

    class Block(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.in_ln = nn.LayerNorm(channels, eps=1e-6)
            self.linears = nn.Sequential(
                nn.Linear(2 * channels, channels), nn.SiLU(),
                nn.Linear(channels, channels), nn.SiLU(),
                nn.Linear(channels, 2 * channels))
            self.gate_act = nn.SiLU()
            self.down_proj = nn.Linear(channels, channels)

        def forward(self, x, y):
            h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
            gate, up = torch.chunk(h, 2, dim=-1)
            return x + self.down_proj(self.gate_act(gate) * up)

    def __init__(self, hidden_size, latent_size=128, noise_size=64, num_mlp_layers=4):
        super().__init__()
        self.noise_size = noise_size
        self.noise_embd = nn.Linear(noise_size, hidden_size)
        self.hidden_embd = nn.Linear(hidden_size, hidden_size)
        self.norm_hidden = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_noise = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp_blocks = nn.ModuleList(
            [self.Block(hidden_size) for _ in range(num_mlp_layers)])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, latent_size))

    def forward(self, hidden_states):
        noise = torch.rand((*hidden_states.shape[:-1], self.noise_size),
                           dtype=hidden_states.dtype,
                           device=hidden_states.device) - 0.5
        h = self.norm_noise(self.noise_embd(noise))
        y = self.norm_hidden(self.hidden_embd(hidden_states))
        for block in self.mlp_blocks:
            h = block(h, y)
        return self.final_layer(h)


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1e3


def main():
    args = preset_args("helm_mice_120M")
    dim, vocab = args.dim, args.vocab_size
    batch, seq = 2, 1024
    num_samples = 8      # CALM's default: the energy score needs N samples per step

    print(f"HELM-MiCE 120M: dim={dim}, vocab={vocab}, {args.n_layers} layers\n")

    vocab_head = nn.Linear(dim, vocab, bias=False)
    calm_head = CalmHead(dim)
    hidden = torch.randn(batch, seq, dim)

    head_params = sum(p.numel() for p in vocab_head.parameters())
    calm_params = sum(p.numel() for p in calm_head.parameters())

    print("Head parameters")
    print(f"  HELM  Linear({dim}, {vocab})       {head_params / 1e6:8.2f} M")
    print(f"  CALM  MLPGenerator -> latent 128  {calm_params / 1e6:8.2f} M"
          f"   ({head_params / calm_params:.0f}x smaller)\n")

    logits_bytes = batch * seq * vocab * 4
    latent_bytes = num_samples * batch * seq * 128 * 4
    print(f"Output activation at batch {batch} x {seq} tokens (float32)")
    print(f"  HELM  logits  {logits_bytes / 2**20:9.1f} MiB")
    print(f"  CALM  latents {latent_bytes / 2**20:9.1f} MiB"
          f"   ({logits_bytes / latent_bytes:.0f}x smaller, {num_samples} energy samples)\n")

    def helm_fwd_bwd():
        vocab_head.zero_grad(set_to_none=True)
        vocab_head(hidden).float().square().mean().backward()

    def calm_fwd_bwd():
        calm_head.zero_grad(set_to_none=True)
        repeated = hidden.unsqueeze(0).expand(num_samples, -1, -1, -1)
        calm_head(repeated).square().mean().backward()

    t_helm, t_calm = bench(helm_fwd_bwd), bench(calm_fwd_bwd)
    print("Head forward + backward, measured")
    print(f"  HELM vocab head          {t_helm:8.1f} ms")
    print(f"  CALM head ({num_samples} samples)     {t_calm:8.1f} ms"
          f"   ({t_helm / t_calm:.2f}x)\n")

    print("Sequence-length effect (patch size K): the backbone sees seq/K positions")
    print(f"  {'K':>3s} {'positions':>10s} {'attn scores':>13s} {'other layers':>13s} "
          f"{'AR steps/2048 tok':>18s}")
    for k in (1, 2, 4, 8):
        positions = seq // k
        print(f"  {k:3d} {positions:10d} {1 / k**2:12.3f}x {1 / k:12.3f}x "
              f"{2048 // k:18d}")

    print("""
Caveats, which are the whole question:

* These are *architecture* numbers. Whether a hyperbolic backbone trains under an
  energy-score objective is an empirical question no arithmetic answers.
* The 128k softmax does not vanish from the system, it moves into the frozen
  autoencoder's decoder -- run on K tokens at a time at generation, and not at
  all during LM training. That is why the LM's head cost collapses.
* CALM needs a separately pretrained autoencoder (75M params, ~15B tokens in the
  paper's setup) before the language model can be trained at all.""")


if __name__ == "__main__":
    main()
