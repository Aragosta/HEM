#!/usr/bin/env python3
"""Stage 0: does CALM's token-chunk autoencoder work at HELM's tokenization?

Three checks, in increasing order of what they prove:

1. **Architecture compatibility** -- build CALM's autoencoder at HELM's
   vocabulary (128256) and confirm the parameter count matches the 75M the paper
   reports, i.e. that HELM's vocabulary is the one CALM was sized for.
2. **Tokenizer identity** -- tokenize real WikiText with the Llama-3 tokenizer
   vendored in the CALM repo, and confirm every id is one HELM's embedding
   accepts. This is the claim that makes the released autoencoder reusable.
3. **Reconstruction** -- train an autoencoder from scratch on that WikiText and
   measure per-token reconstruction accuracy against patch size K.

Check 3 is a *proxy*, and its limits should be stated plainly: the released model
is 75M parameters trained on ~15B tokens and reports >99.9% reconstruction. This
machine has no GPU and no access to Hugging Face (blocked), so what is trained
here is far smaller, on ~1 MB of text, over a frequency-truncated vocabulary. It
establishes that the mechanism works and shows how accuracy degrades with K; it
does not validate the released checkpoint.

Usage:
    python CALM/experiments/stage0_autoencoder.py --calm-repo /path/to/calm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HELM_VOCAB_SIZE = 128256


# --------------------------------------------------------------------- check 1

def check_architecture(calm_repo: Path):
    """Build CALM's autoencoder at HELM's vocabulary and count parameters."""
    sys.path.insert(0, str(calm_repo))
    from models.configuration_autoencoder import AutoencoderConfig
    from models.modeling_autoencoder import Autoencoder

    config = AutoencoderConfig(
        vocab_size=HELM_VOCAB_SIZE, hidden_size=512, latent_size=128,
        patch_size=4, num_encoder_layers=2, num_decoder_layers=2)
    model = Autoencoder(config)
    total = sum(p.numel() for p in model.parameters())
    embed = model.encoder.embed_tokens.weight.numel()

    print("1. Architecture compatibility")
    print(f"   CALM autoencoder at vocab_size={HELM_VOCAB_SIZE}, hidden=512, K=4")
    print(f"     total parameters      {total / 1e6:7.1f} M   (paper reports 75M)")
    print(f"     of which embedding    {embed / 1e6:7.1f} M   ({100 * embed / total:.0f}%)")
    print(f"     latent size           {config.latent_size}")
    match = abs(total / 1e6 - 75) < 10
    print(f"     -> {'matches' if match else 'DOES NOT match'} the published size\n")
    return match


# --------------------------------------------------------------------- check 2

def check_tokenizer(calm_repo: Path, n_docs: int = 200):
    """Tokenize real WikiText with CALM's vendored Llama-3 tokenizer."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(calm_repo / "llama3_tokenizer" / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    data_file = calm_repo / "data" / "wikitext_document_level-test.json"
    texts = []
    with open(data_file) as handle:
        for line in handle:
            if len(texts) >= n_docs:
                break
            texts.append(json.loads(line)["text"])

    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False).ids)

    print("2. Tokenizer identity")
    print(f"   CALM's vendored tokenizer vocab   {vocab_size}")
    print(f"   HELM's configured vocab_size      {HELM_VOCAB_SIZE}")
    print(f"   tokenized {len(texts)} WikiText documents -> {len(ids):,} tokens")
    print(f"   id range observed                 [{min(ids)}, {max(ids)}]")
    ok = vocab_size == HELM_VOCAB_SIZE and max(ids) < HELM_VOCAB_SIZE
    print(f"   -> HELM's embedding {'accepts every id' if ok else 'WOULD OVERFLOW'}\n")
    return ok, ids


# --------------------------------------------------------------------- check 3

class TinyAutoencoder(nn.Module):
    """CALM's autoencoder shape, scaled down to something CPU-trainable.

    Same structure as ``upstream/models/modeling_autoencoder.py``: embed K
    tokens, MLP blocks, squeeze to one vector, project to a latent; then expand
    back and decode through a tied head. Attention-free, exactly as upstream.
    """

    class Block(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.norm = nn.RMSNorm(hidden, eps=1e-5)
            self.gate = nn.Linear(hidden, 4 * hidden, bias=False)
            self.up = nn.Linear(hidden, 4 * hidden, bias=False)
            self.down = nn.Linear(4 * hidden, hidden, bias=False)

        def forward(self, x):
            h = self.norm(x)
            return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def __init__(self, vocab, hidden=256, latent=128, patch=4, layers=2):
        super().__init__()
        self.patch, self.latent_size = patch, latent
        self.embed = nn.Embedding(vocab, hidden)
        self.enc_a = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.squeeze = nn.Linear(patch * hidden, hidden)
        self.enc_b = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.enc_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.to_latent = nn.Linear(hidden, latent * 2)

        self.from_latent = nn.Linear(latent, hidden)
        self.dec_a = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.expand = nn.Linear(hidden, patch * hidden)
        self.dec_b = nn.ModuleList([self.Block(hidden) for _ in range(layers)])
        self.dec_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight          # tied, as upstream

    def encode(self, ids):
        """(B, K) token ids -> (mean, log_std), each (B, latent)."""
        h = self.embed(ids)
        for block in self.enc_a:
            h = block(h)
        h = self.squeeze(h.flatten(1))
        for block in self.enc_b:
            h = block(h)
        return self.to_latent(self.enc_norm(h)).chunk(2, dim=-1)

    def decode(self, latent):
        """(B, latent) -> (B, K, vocab) logits."""
        h = self.from_latent(latent)
        for block in self.dec_a:
            h = block(h)
        h = self.expand(h).view(latent.size(0), self.patch, -1)
        for block in self.dec_b:
            h = block(h)
        return self.head(self.dec_norm(h))

    def forward(self, ids, kl_weight=1e-3):
        mean, log_std = self.encode(ids)
        latent = mean + torch.randn_like(mean) * log_std.exp()
        logits = self.decode(latent)
        recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
        kl = (0.5 * (mean.pow(2) + (2 * log_std).exp() - 1) - log_std).sum(-1).mean()
        return recon * self.patch + kl_weight * kl, recon, logits


def train_autoencoder(ids, patch, steps=600, batch=64, hidden=256, top_k_vocab=4096,
                      seed=0, log_every=200):
    """Train a small autoencoder on real text; return held-out token accuracy."""
    torch.manual_seed(seed)

    # Frequency-truncate the vocabulary so the tied head is CPU-tractable. The
    # task is easier than the full 128k vocabulary -- stated, not hidden.
    counts = Counter(ids)
    keep = [tok for tok, _ in counts.most_common(top_k_vocab - 1)]
    remap = {tok: i + 1 for i, tok in enumerate(keep)}       # 0 = out-of-vocab
    compact = torch.tensor([remap.get(t, 0) for t in ids], dtype=torch.long)
    coverage = sum(counts[t] for t in keep) / len(ids)

    usable = (compact.numel() // patch) * patch
    patches = compact[:usable].view(-1, patch)
    split = int(0.9 * patches.size(0))
    train_set, eval_set = patches[:split], patches[split:]

    model = TinyAutoencoder(top_k_vocab, hidden=hidden, patch=patch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    model.train()
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        rows = torch.randint(0, train_set.size(0), (batch,), generator=generator)
        loss, recon, _ = model(train_set[rows])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0 or step == 1:
            print(f"      step {step:4d}  recon-CE {recon.item():.3f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start_idx in range(0, min(eval_set.size(0), 2048), 256):
            chunk = eval_set[start_idx:start_idx + 256]
            mean, _ = model.encode(chunk)                    # deterministic at eval
            predicted = model.decode(mean).argmax(-1)
            correct += (predicted == chunk).sum().item()
            total += chunk.numel()
    return correct / total, coverage, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calm-repo", type=Path, required=True,
                        help="clone of github.com/shaochenze/calm (for tokenizer + data)")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--patches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--skip-architecture", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("STAGE 0 -- CALM autoencoder vs HELM tokenization")
    print("=" * 72 + "\n")

    arch_ok = True
    if not args.skip_architecture:
        arch_ok = check_architecture(args.calm_repo)
    tok_ok, ids = check_tokenizer(args.calm_repo)

    print("3. Reconstruction from scratch on real WikiText")
    print("   (proxy only: the released model is 75M params on ~15B tokens and")
    print("    reports >99.9%; Hugging Face is unreachable from this machine)\n")
    results = {}
    for patch in args.patches:
        print(f"   K = {patch}")
        accuracy, coverage, seconds = train_autoencoder(ids, patch, steps=args.steps)
        results[patch] = accuracy
        print(f"      -> token reconstruction accuracy {accuracy:6.2%}"
              f"   ({seconds:.0f}s, vocab coverage {coverage:.1%})\n")

    print("=" * 72)
    print(f"architecture compatible : {arch_ok}")
    print(f"tokenizer compatible    : {tok_ok}")
    for patch, accuracy in results.items():
        print(f"reconstruction K={patch}     : {accuracy:.2%}")
    print("=" * 72)


if __name__ == "__main__":
    main()
