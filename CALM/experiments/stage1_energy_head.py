#!/usr/bin/env python3
"""Stage 1: does a *hyperbolic* backbone train under CALM's energy score?

The one question worth answering before building anything else. CALM's
likelihood-free objective has only ever been run on Euclidean transformers.
HELM's activations are constrained to a Lorentz hyperboloid -- every layer
renormalises onto the manifold -- and the interaction with a sampling-based
objective is unexplored.

The experiment isolates that single variable. Two models share the *same*
HELM-MiCE backbone, the same data, the same optimizer and the same number of
steps. They differ only in the head:

* **discrete** -- HELM's ``Linear(dim, vocab)`` and cross-entropy (the baseline);
* **CALM** -- CALM's ``MLPGenerator`` predicting a continuous latent, trained
  with the energy score against a frozen K=1 autoencoder's posterior.

Patch size is 1 deliberately: no compression, no sequence-length change, nothing
that could confound "the objective works" with "the compression works". At K=1
there is no efficiency win to be had -- that is not what this measures.

Both are scored by the same metric, next-token accuracy, which the CALM model
reaches by decoding its predicted latent through the frozen autoencoder.

Usage:
    python CALM/experiments/stage1_energy_head.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from tests._config import tiny_args  # noqa: E402


# ----------------------------------------------------------------- autoencoder

class PatchAutoencoder(nn.Module):
    """CALM's autoencoder at patch size 1, small enough to pretrain in seconds.

    Encodes a token to a Gaussian posterior over a continuous latent and decodes
    it back. Same shape as ``upstream/models/modeling_autoencoder.py``: MLP
    blocks, a latent bottleneck carrying ``(mean, log_std)``, and a tied head.
    """

    class Block(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.norm = nn.RMSNorm(hidden, eps=1e-5)
            self.gate = nn.Linear(hidden, 2 * hidden, bias=False)
            self.up = nn.Linear(hidden, 2 * hidden, bias=False)
            self.down = nn.Linear(2 * hidden, hidden, bias=False)

        def forward(self, x):
            h = self.norm(x)
            return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def __init__(self, vocab, hidden=128, latent=32, layers=2):
        super().__init__()
        self.latent_size = latent
        self.embed = nn.Embedding(vocab, hidden)
        self.enc = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.enc_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.to_latent = nn.Linear(hidden, latent * 2)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.dec_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight

    def encode(self, ids):
        """token ids -> (mean, log_std)."""
        h = self.embed(ids)
        for block in self.enc:
            h = block(h)
        return self.to_latent(self.enc_norm(h)).chunk(2, dim=-1)

    def decode(self, latent):
        """latent -> vocab logits."""
        h = self.from_latent(latent)
        for block in self.dec:
            h = block(h)
        return self.head(self.dec_norm(h))


def pretrain_autoencoder(vocab, batches, steps=400, kl_weight=1e-3, seed=0):
    """Train the K=1 autoencoder to convergence; it is frozen thereafter."""
    torch.manual_seed(seed)
    model = PatchAutoencoder(vocab)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        tokens = batches[torch.randint(0, len(batches), (1,), generator=generator)].reshape(-1)
        mean, log_std = model.encode(tokens)
        latent = mean + torch.randn_like(mean) * log_std.exp()
        logits = model.decode(latent)
        recon = F.cross_entropy(logits, tokens)
        kl = (0.5 * (mean.pow(2) + (2 * log_std).exp() - 1) - log_std).sum(-1).mean()
        optimizer.zero_grad()
        (recon + kl_weight * kl).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tokens = batches.reshape(-1)
        mean, _ = model.encode(tokens)
        accuracy = (model.decode(mean).argmax(-1) == tokens).float().mean().item()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, accuracy


# ------------------------------------------------------------------- CALM head

class CalmHead(nn.Module):
    """CALM's ``MLPGenerator``: noise + hidden state -> continuous latent.

    Transcribed from ``upstream/models/modeling_energy.py``.
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

    def __init__(self, hidden_size, latent_size, noise_size=64, num_mlp_layers=4):
        super().__init__()
        self.noise_size = noise_size
        self.noise_embd = nn.Linear(noise_size, hidden_size)
        self.hidden_embd = nn.Linear(hidden_size, hidden_size)
        self.norm_noise = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_hidden = nn.LayerNorm(hidden_size, eps=1e-6)
        self.blocks = nn.ModuleList([self.Block(hidden_size) for _ in range(num_mlp_layers)])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, latent_size))
        # CALM zero-initialises the final projection so the head starts neutral.
        nn.init.zeros_(self.final_layer[-1].weight)
        nn.init.zeros_(self.final_layer[-1].bias)

    def sample(self, hidden_states):
        noise = torch.rand((*hidden_states.shape[:-1], self.noise_size),
                           dtype=hidden_states.dtype,
                           device=hidden_states.device) - 0.5
        h = self.norm_noise(self.noise_embd(noise))
        y = self.norm_hidden(self.hidden_embd(hidden_states))
        for block in self.blocks:
            h = block(h, y)
        return self.final_layer(h)


def energy_score(samples, mean, log_std, beta=1.0, n_target=100):
    """CALM's energy score, transcribed from ``modeling_energy.py``.

    A strictly proper scoring rule for beta in (0, 2): rewards samples close to
    the target distribution while penalising a collapsed predictive distribution.
    """
    def distance(a, b):
        return torch.linalg.norm(a - b, ord=2, dim=-1).pow(beta)

    n_x = samples.shape[0]
    pairwise = distance(samples.unsqueeze(1), samples.unsqueeze(0))
    distance_x = pairwise.sum(dim=(0, 1)) / (n_x * (n_x - 1))

    std = log_std.exp()
    targets = mean + torch.randn((n_target, *mean.shape), device=mean.device,
                                 dtype=mean.dtype) * std
    cross = distance(samples.reshape(n_x, 1, *samples.shape[1:]),
                     targets.reshape(1, n_target, *targets.shape[1:]))
    return distance_x - cross.mean(dim=(0, 1)) * 2


class EuclideanBackbone(nn.Module):
    """A minimal Euclidean pre-norm transformer, matched to HELM's shape.

    The control. Without it, a CALM head that fails to learn on HELM tells you
    nothing: it could be the manifold, or it could be the step budget, the
    learning rate, or the objective's variance. Running the identical head and
    objective on a Euclidean backbone of the same width and depth separates
    those.
    """

    class Layer(nn.Module):
        def __init__(self, dim, heads, inter):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.RMSNorm(dim, eps=1e-5)
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)
            self.norm2 = nn.RMSNorm(dim, eps=1e-5)
            self.gate = nn.Linear(dim, inter, bias=False)
            self.up = nn.Linear(dim, inter, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)

        def forward(self, x):
            b, n, d = x.shape
            q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
            shape = (b, n, self.heads, d // self.heads)
            q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.out(attn.transpose(1, 2).reshape(b, n, d))
            h = self.norm2(x)
            return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def __init__(self, vocab, dim, n_layers, n_heads, inter):
        super().__init__()
        # `dim` must divide by heads; HELM's odd dim does not, so round up.
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, 512, dim))
        self.layers = nn.ModuleList(
            [self.Layer(dim, n_heads, inter) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim, eps=1e-5)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos[:, :tokens.size(1)]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ----------------------------------------------------------------------- data

def make_batches(vocab, n_batches, batch=2, length=24, seed=1234):
    """An arithmetic walk: t_{i+1} = (t_i + stride) mod vocab. Learnable, cheap."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        start = torch.randint(0, vocab, (batch, 1), generator=generator)
        stride = torch.randint(1, 4, (batch, 1), generator=generator)
        offsets = torch.arange(length).unsqueeze(0)
        batches.append((start + stride * offsets) % vocab)
    return torch.stack(batches)


# ------------------------------------------------------------------ experiment

def backbone_hidden(model, tokens):
    """HELM's final hidden state -- the exact tensor its vocab head consumes."""
    seqlen = tokens.size(-1)
    h = model.embed(tokens)
    freqs_cis = model.freqs_cis[:seqlen]
    for layer in model.layers:
        out = layer(h, 0, freqs_cis, None, True, None)
        h = out[0] if isinstance(out, tuple) else out
    return model.norm(model.final_proj(h, return_space=True), space_only=True)


def run_discrete(args, batches, steps, lr, seed):
    """Baseline: HELM as published, cross-entropy over the vocabulary."""
    torch.manual_seed(seed)
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    losses, start = [], time.perf_counter()
    for step in range(steps):
        tokens = batches[step % len(batches)]
        logits = model(tokens)[0]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               tokens[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for tokens in batches:
            predicted = model(tokens)[:, :-1].argmax(-1)
            correct += (predicted == tokens[:, 1:]).sum().item()
            total += tokens[:, 1:].numel()
    return losses, correct / total, time.perf_counter() - start


def run_calm(args, batches, autoencoder, steps, lr, seed, num_samples=8, beta=1.0,
             euclidean=False):
    """A backbone with CALM's generative head and energy-score objective.

    ``euclidean=True`` swaps HELM for :class:`EuclideanBackbone` and changes
    nothing else -- the control described in that class's docstring.
    """
    torch.manual_seed(seed)
    if euclidean:
        model = EuclideanBackbone(args.vocab_size, args.dim + 3, args.n_layers,
                                  args.n_heads, args.inter_dim)
        width = args.dim + 3
        hidden_of = lambda toks: model(toks)
    else:
        model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
        del model.head                                 # the vocab projection goes
        width = args.dim
        hidden_of = lambda toks: backbone_hidden(model, toks)
    head = CalmHead(width, autoencoder.latent_size)

    parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    model.train()
    head.train()

    losses, start = [], time.perf_counter()
    for step in range(steps):
        tokens = batches[step % len(batches)]
        targets = tokens[:, 1:].reshape(-1)
        with torch.no_grad():
            mean, log_std = autoencoder.encode(targets)

        hidden = hidden_of(tokens)[:, :-1].reshape(-1, width)
        repeated = hidden.unsqueeze(0).expand(num_samples, -1, -1)
        samples = head.sample(repeated)
        loss = -energy_score(samples, mean, log_std, beta=beta).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for tokens in batches:
            targets = tokens[:, 1:].reshape(-1)
            hidden = hidden_of(tokens)[:, :-1].reshape(-1, width)
            repeated = hidden.unsqueeze(0).expand(num_samples, -1, -1)
            decoded = autoencoder.decode(head.sample(repeated)).argmax(-1)
            # Majority vote over the sample pool approximates the mode, which is
            # what CALM's temperature-0 sampling targets.
            voted = torch.mode(decoded, dim=0).values
            correct += (voted == targets).sum().item()
            total += targets.numel()
    return losses, correct / total, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    cli = parser.parse_args()

    args = tiny_args()
    batches = make_batches(args.vocab_size, n_batches=16)

    print("=" * 72)
    print("STAGE 1 -- does a hyperbolic backbone train under an energy score?")
    print("=" * 72)
    print(f"\nbackbone: HELM-MiCE dim={args.dim}, {args.n_layers} layers, "
          f"{args.n_routed_experts} experts, vocab={args.vocab_size}")
    print(f"task: arithmetic walk, {cli.steps} steps, lr={cli.lr}, patch size K=1\n")

    autoencoder, ae_accuracy = pretrain_autoencoder(args.vocab_size, batches)
    print(f"frozen K=1 autoencoder: reconstruction {ae_accuracy:.2%}, "
          f"latent {autoencoder.latent_size}d")
    if ae_accuracy < 0.99:
        print("  WARNING: autoencoder has not converged; Stage 1 is not "
              "interpretable until it has")

    print("\n--- discrete HELM (cross-entropy over vocabulary) ---")
    d_losses, d_accuracy, d_time = run_discrete(args, batches, cli.steps, cli.lr, cli.seed)
    print(f"  loss {d_losses[0]:8.3f} -> {d_losses[-1]:8.3f}   "
          f"next-token accuracy {d_accuracy:6.2%}   ({d_time:.0f}s)")

    print("\n--- CALM head on a EUCLIDEAN backbone (the control) ---")
    e_losses, e_accuracy, e_time = run_calm(args, batches, autoencoder, cli.steps,
                                            cli.lr, cli.seed, cli.num_samples,
                                            euclidean=True)
    print(f"  loss {e_losses[0]:8.3f} -> {e_losses[-1]:8.3f}   "
          f"next-token accuracy {e_accuracy:6.2%}   ({e_time:.0f}s)")

    print("\n--- CALM head on the HELM backbone (energy score) ---")
    c_losses, c_accuracy, c_time = run_calm(args, batches, autoencoder, cli.steps,
                                            cli.lr, cli.seed, cli.num_samples)
    print(f"  loss {c_losses[0]:8.3f} -> {c_losses[-1]:8.3f}   "
          f"next-token accuracy {c_accuracy:6.2%}   ({c_time:.0f}s)")

    chance = 1.0 / args.vocab_size
    threshold = 10 * chance
    print("\n" + "=" * 72)
    print(f"chance accuracy                     {chance:6.2%}")
    print(f"discrete HELM (cross-entropy)       {d_accuracy:6.2%}")
    print(f"CALM head, Euclidean backbone       {e_accuracy:6.2%}   <- control")
    print(f"CALM head, HELM backbone            {c_accuracy:6.2%}")

    helm_learns = c_accuracy > threshold
    control_learns = e_accuracy > threshold
    print()
    if control_learns and helm_learns:
        print("-> the energy score trains BOTH backbones. Hyperbolic geometry is")
        print("   not an obstacle to CALM's objective at this scale.")
    elif control_learns and not helm_learns:
        print("-> the energy score trains the Euclidean backbone but NOT the")
        print("   hyperbolic one. This is the real negative result: the manifold")
        print("   constraint is interacting badly with the objective.")
    elif not control_learns:
        print("-> the control did not learn either, so this run says nothing about")
        print("   hyperbolic geometry -- the step budget or learning rate is the")
        print("   binding constraint. Re-run with more steps before concluding.")
    print("=" * 72)


if __name__ == "__main__":
    main()
