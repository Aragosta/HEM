#!/usr/bin/env python3
"""Real English text, real patches, and the metrics a language-model paper reports.

``EVALUATION.md`` sets out why nothing else in ``experiments/`` counts as a
language-modelling result: a synthetic tree grammar, 1024 tokens, ``K = 1``, and
until recently no held-out split. This script fixes as much of that as this
environment allows.

**Corpus.** The repository's own English prose and source -- every ``.md`` and
``.py`` outside ``upstream/`` -- read as **bytes**, vocabulary 256. Byte level is
not a compromise here: it removes the tokenizer from the comparison entirely,
which is why bits-per-byte is the tokenizer-independent metric labs report when
models do not share a vocabulary. The split is **by file**, not by slicing inside
a document, so no validation context has its own neighbourhood in the training
set.

The honest limitation: this is ~900 KB, which is four to six orders of magnitude
below a real pretraining corpus, and HuggingFace is unreachable from this
environment (the egress proxy rejects it) so WikiText-103 and the Llama-3
tokenizer cannot be fetched. What this buys over the tree grammar is real
Zipfian statistics, real long-range structure, and a genuine train/validation
separation. What it does not buy is scale.

**Metrics.**

``bits_per_byte``
    ``-log2 p(byte)`` averaged over held-out bytes. The standard, and the one
    number here directly comparable to published work in kind (not in value,
    at this scale). Available for the **discrete** model only: HELM-CALM's head
    is an implicit sampler with no density, which is a property of CALM's design,
    not a gap in this port.

``BrierLM``
    CALM's likelihood-free proper score, from ``brierlm.py``. Computable for
    both model families, and therefore the only metric on which they can be
    compared at all.

``K = 4`` throughout, so the patching that is CALM's entire reason to exist is
actually exercised.

Usage::

    python CALM/experiments/real_text.py --steps 3000
    python CALM/experiments/real_text.py --steps 3000 --arms discrete,euclidean
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments"):
    sys.path.insert(0, str(_extra))

import torch.nn as nn  # noqa: E402

from brierlm import brier_lm, brier_scores  # noqa: E402
from stage1_energy_head import EuclideanBackbone  # noqa: E402
from helm_calm import CalmEnergyHead  # noqa: E402
from helm_calm import HelmCALM, PatchAutoencoder, energy_score  # noqa: E402
from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from hyperbolic_latent import LorentzPatchAutoencoder  # noqa: E402
from tests._config import tiny_args  # noqa: E402

VOCAB = 256


# ------------------------------------------------------------------- corpus

def collect_files() -> List[Path]:
    """Every ``.md`` and ``.py`` in the repo that is our own writing."""
    files = []
    for pattern in ("*.md", "*.py"):
        for path in sorted(ROOT.rglob(pattern)):
            parts = set(path.parts)
            if parts & {".git", "upstream", "__pycache__", "node_modules"}:
                continue
            files.append(path)
    return files


def build_corpus(valid_fraction: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Byte tensors for train and validation, split by whole file.

    Splitting by file rather than by offset is the point: adjacent slices of one
    document share vocabulary, topic and often literal substrings, so an
    offset split leaks and flatters the validation number.
    """
    files = collect_files()
    # Deterministic, and interleaved so both halves see every directory rather
    # than the validation set being whatever sorts last.
    stride = max(int(1 / valid_fraction), 2)
    train_bytes, valid_bytes = bytearray(), bytearray()
    for index, path in enumerate(files):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        (valid_bytes if index % stride == 0 else train_bytes).extend(raw)
    return (torch.frombuffer(bytes(train_bytes), dtype=torch.uint8).long(),
            torch.frombuffer(bytes(valid_bytes), dtype=torch.uint8).long())


def batches_from(data: torch.Tensor, batch_size: int, seq_len: int, count: int,
                 seed: int) -> List[torch.Tensor]:
    """Fixed set of random windows, so every arm sees identical data."""
    generator = torch.Generator().manual_seed(seed)
    highest = data.numel() - seq_len - 1
    out = []
    for _ in range(count):
        starts = torch.randint(0, highest, (batch_size,), generator=generator)
        out.append(torch.stack([data[s:s + seq_len] for s in starts]))
    return out


# ------------------------------------------------------------------ controls

class EuclideanCalm(nn.Module):
    """HELM-CALM with the geometry removed and nothing else changed.

    This is the arm that answers "does hyperbolic geometry help or hurt under
    CALM's objective". It mirrors :class:`~CALM.helm_calm.HelmCALM` piece for
    piece -- patch embedding, backbone of the same width and depth, CALM's
    generative head, the same frozen autoencoder, the same energy score, the
    same optimizer and budget -- and differs *only* in that every component is
    flat. Any difference between this and the HELM arm is attributable to the
    manifold, and to nothing else.

    Matched at ``dim - 1``: a Lorentz vector of width ``dim`` carries ``dim - 1``
    free coordinates, since the time coordinate is determined by the rest. Giving
    the Euclidean control ``dim`` features would hand it one extra free parameter
    per position and turn a geometry comparison into a capacity comparison.
    """

    def __init__(self, vocab_size, dim, layers, heads, inter, patch, latent):
        super().__init__()
        self.patch_size = patch
        self.width = dim - 1
        # HELM's head count is chosen for HMLA's latent shapes and need not
        # divide dim - 1. Plain multi-head attention does require that, so pick
        # the nearest divisor. Head count is not the variable under test.
        if self.width % heads:
            divisors = [h for h in range(1, self.width + 1) if self.width % h == 0]
            heads = min(divisors, key=lambda h: (abs(h - heads), h))
        self.embed = nn.Embedding(vocab_size, self.width)
        self.patch_embed = nn.Linear(patch * self.width, self.width)
        # Reuse the control backbone's layers, but feed them patch vectors
        # directly instead of token ids -- its own embedding is left unused
        # rather than modifying stage1_energy_head, which other experiments read.
        self.backbone = EuclideanBackbone(vocab_size, self.width, layers, heads,
                                          inter)
        del self.backbone.embed
        self.head = CalmEnergyHead(self.width, latent)

    def hidden_states(self, tokens):
        embedded = self.embed(tokens)
        b, s, _ = embedded.shape
        x = self.patch_embed(embedded.reshape(b, s // self.patch_size, -1))
        x = x + self.backbone.pos[:, :x.size(1)]
        for layer in self.backbone.layers:
            x = layer(x)
        return self.backbone.norm(x)

    def _aligned(self, tokens):
        n_patches = tokens.size(1) // self.patch_size
        inputs = tokens[:, :(n_patches - 1) * self.patch_size]
        targets = tokens[:, self.patch_size:n_patches * self.patch_size]
        return inputs, targets.reshape(-1, self.patch_size)


# --------------------------------------------------------------------- arms

def model_args(seq_len: int) -> object:
    return tiny_args(vocab_size=VOCAB, dim=65, n_layers=4, n_heads=5,
                     max_seq_len=seq_len, original_seq_len=seq_len,
                     max_batch_size=8, qk_nope_head_dim=13, qk_rope_head_dim=13,
                     v_head_dim=13, kv_lora_rank=33, inter_dim=128,
                     moe_inter_dim=96, mice_inter_dim=96)


def train_discrete(args, train_batches, steps, lr):
    """Optimized HELM-MiCE with its ordinary vocabulary head."""
    torch.manual_seed(0)
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    for step in range(steps):
        tokens = train_batches[step % len(train_batches)]
        out = model(tokens[:, :-1], labels=tokens[:, 1:])
        loss = out[0] if isinstance(out, tuple) else out
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def bits_per_byte(model, batches) -> float:
    """Held-out ``-log2 p(byte)``. The metric, for the model that has a density."""
    total_nats, total_bytes = 0.0, 0
    for tokens in batches:
        out = model(tokens[:, :-1], labels=tokens[:, 1:])
        loss = out[0] if isinstance(out, tuple) else out
        count = tokens[:, 1:].numel()
        total_nats += loss.item() * count
        total_bytes += count
    return total_nats / total_bytes / math.log(2)


@torch.no_grad()
def discrete_draw(model, tokens, n):
    """``(n, B, S-1)`` independent teacher-forced draws."""
    out = model(tokens[:, :-1])
    logits = (out[0] if isinstance(out, tuple) else out).float()
    probs = logits.softmax(-1)
    flat = probs.reshape(-1, probs.size(-1))
    drawn = torch.multinomial(flat, n, replacement=True).T
    return drawn.reshape(n, *probs.shape[:-1])


@torch.no_grad()
def discrete_samples(model, tokens):
    """Two independent teacher-forced draws, ``(B, S-1)`` each, plus targets.

    BrierLM scores n-grams up to order 4, so the sequence axis has to survive:
    flattening to a single vector would make every "n-gram" a splice across
    unrelated positions.
    """
    out = model(tokens[:, :-1])
    logits = (out[0] if isinstance(out, tuple) else out).float()
    probs = logits.softmax(-1)
    flat = probs.reshape(-1, probs.size(-1))
    first = torch.multinomial(flat, 1).reshape(probs.shape[:-1])
    second = torch.multinomial(flat, 1).reshape(probs.shape[:-1])
    return first, second, tokens[:, 1:]


def train_autoencoder(cls, train_data, patch, latent, steps, seed=0):
    torch.manual_seed(seed)
    autoencoder = cls(VOCAB, hidden=128, latent_size=latent, patch_size=patch)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)
    usable = (train_data.numel() // patch) * patch
    patches = train_data[:usable].view(-1, patch)
    for _ in range(steps):
        rows = torch.randint(0, patches.size(0), (128,), generator=generator)
        loss, _ = autoencoder.elbo(patches[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return autoencoder.freeze()


@torch.no_grad()
def autoencoder_ceiling(autoencoder, data) -> float:
    patch = autoencoder.patch_size
    usable = (data.numel() // patch) * patch
    patches = data[:usable].view(-1, patch)[:4096]
    posterior = autoencoder.encode(patches)
    latent = (posterior.mean if getattr(autoencoder, "is_hyperbolic", False)
              else posterior[0])
    return (autoencoder.decode(latent).argmax(-1) == patches).float().mean().item()


def train_calm(args, autoencoder, train_batches, steps, lr):
    torch.manual_seed(0)
    model = HelmCALM(args, autoencoder, num_samples=8, head_kind="lorentz")
    groups = model.parameter_groups()
    params = groups["euclidean"] + groups["manifold"]
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    for step in range(steps):
        loss = model.loss(train_batches[step % len(train_batches)])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        model.retract_manifold_parameters()
    return model.eval()


def train_euclidean_calm(args, autoencoder, train_batches, steps, lr, patch):
    torch.manual_seed(0)
    model = EuclideanCalm(VOCAB, args.dim, args.n_layers, args.n_heads,
                          args.inter_dim, patch, autoencoder.latent_size)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    for step in range(steps):
        tokens = train_batches[step % len(train_batches)]
        inputs, targets = model._aligned(tokens)
        with torch.no_grad():
            mean, log_std = autoencoder.encode(targets)
        hidden = model.hidden_states(inputs).reshape(-1, model.width)
        samples = model.head(hidden.unsqueeze(0).expand(8, -1, -1))
        loss = -energy_score(samples, mean, log_std).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def euclidean_calm_draw(model, autoencoder, tokens, n):
    """``(n, B, S')`` byte draws plus targets -- the same shape the HELM arm gives."""
    inputs, targets = model._aligned(tokens)
    hidden = model.hidden_states(inputs).reshape(-1, model.width)
    latents = model.head(hidden.unsqueeze(0).expand(n, -1, -1))
    decoded = autoencoder.decode(latents).argmax(-1)
    rows = tokens.size(0)
    return decoded.reshape(n, rows, -1), targets.reshape(rows, -1)


# ------------------------------------------------------------------ baselines

def byte_baselines(train_data, valid_batches):
    """What a lookup table achieves on the same held-out bytes.

    The lesson from the tree-language run: a neural accuracy is uninterpretable
    without this. A model that cannot beat a bigram table has not learned the
    corpus, whatever its training loss says.
    """
    import collections
    counts = collections.Counter(train_data.tolist())
    mode = counts.most_common(1)[0][0]
    following = collections.defaultdict(collections.Counter)
    previous = train_data[:-1].tolist()
    nxt = train_data[1:].tolist()
    for a, b in zip(previous, nxt):
        following[a][b] += 1
    best = {k: v.most_common(1)[0][0] for k, v in following.items()}

    # Bits per byte for the smoothed bigram model, so the neural BPB has a
    # reference on its own scale as well.
    vocab_totals = {k: sum(v.values()) for k, v in following.items()}
    correct = total = 0
    nats = 0.0
    for tokens in valid_batches:
        for row in tokens:
            ids = row.tolist()
            for a, b in zip(ids[:-1], ids[1:]):
                total += 1
                correct += (best.get(a, mode) == b)
                table = following.get(a)
                count = (table.get(b, 0) if table else 0)
                denominator = (vocab_totals.get(a, 0) + VOCAB)
                nats -= math.log((count + 1) / denominator)
    return {"bigram top-1": correct / total,
            "bigram BPB": nats / total / math.log(2),
            "unigram mode top-1": sum(
                (row == mode).sum().item() for t in valid_batches for row in t)
            / sum(t.numel() for t in valid_batches)}


# ------------------------------------------------------------------ scoring

def top1_accuracy(draw, batches, n_samples=32) -> float:
    """Modal prediction accuracy, in byte space, identical across all arms.

    This is the number that can sit next to the bigram baseline, because it is
    measured on the same quantity: which byte comes next.
    """
    correct = total = 0
    for tokens in batches:
        samples, targets = draw(tokens, n_samples)
        predicted = torch.mode(samples, dim=0).values
        correct += (predicted == targets).sum().item()
        total += targets.numel()
    return correct / total


def score_brierlm(sampler, batches, max_n=4):
    """``sampler`` returns two independent draws and the targets, sequence-shaped.

    Returns ``(per_order, brierlm)``.

    **Why the per-order scores are reported and not just the aggregate.**
    BrierLM is a geometric mean over orders 1..4, so a single zero factor sends
    it to exactly zero. At byte level two independent draws agree on a 4-gram
    with probability around 256^-4 unless the model is genuinely strong, so
    ``brier_4`` pins to 0 and the aggregate reads 0.0000 for *every* arm --
    identical output for a good model and a random one. The aggregate is still
    printed, because it is what CALM reports, but it is not the comparison; the
    per-order scores are.
    """
    totals = torch.zeros(max_n, dtype=torch.float64)
    aggregate = 0.0
    for tokens in batches:
        first, second, targets = sampler(tokens)
        totals += brier_scores(first, second, targets, max_n).double()
        aggregate += brier_lm(first, second, targets, max_n=max_n)
    return (totals / len(batches)).tolist(), aggregate / len(batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--ae-steps", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--arms", default="discrete,helm-calm,euclidean-calm,hyperbolic")
    args_cli = parser.parse_args()
    arms = set(args_cli.arms.split(","))

    train_data, valid_data = build_corpus()
    print(f"corpus: {train_data.numel():,} train bytes, "
          f"{valid_data.numel():,} validation bytes, split by file")
    print(f"K = {args_cli.patch}, seq_len = {args_cli.seq_len}, "
          f"{args_cli.steps} steps\n")

    train_batches = batches_from(train_data, args_cli.batch_size,
                                 args_cli.seq_len, 64, seed=0)
    valid_batches = batches_from(valid_data, args_cli.batch_size,
                                 args_cli.seq_len, 16, seed=1)
    args = model_args(args_cli.seq_len)
    results = {}

    baselines = byte_baselines(train_data, valid_batches)
    print("baselines on the same held-out bytes")
    print(f"  bigram lookup table   top-1 {baselines['bigram top-1']:.2%}   "
          f"BPB {baselines['bigram BPB']:.4f}")
    print(f"  unigram mode          top-1 {baselines['unigram mode top-1']:.2%}")
    print(f"  uniform               top-1 {1 / VOCAB:.2%}   BPB 8.0000\n")

    if "discrete" in arms:
        model = train_discrete(args, train_batches, args_cli.steps, args_cli.lr)
        results["discrete HELM"] = {
            "bpb": bits_per_byte(model, valid_batches),
            "bpb_train": bits_per_byte(model, train_batches[:16]),
            "top1": top1_accuracy(
                lambda t, n: (discrete_draw(model, t, n), t[:, 1:]),
                valid_batches),
        }
        orders, aggregate = score_brierlm(
            lambda t: discrete_samples(model, t), valid_batches)
        results["discrete HELM"].update(brier_orders=orders, brierlm=aggregate)

    # One autoencoder per latent geometry, shared by the arms that use it, so
    # the HELM and Euclidean CALM arms differ only in the backbone.
    autoencoders = {}
    if arms & {"helm-calm", "euclidean-calm"}:
        autoencoders["euclidean"] = train_autoencoder(
            PatchAutoencoder, train_data, args_cli.patch, args_cli.latent,
            args_cli.ae_steps)
    if "hyperbolic" in arms:
        autoencoders["hyperbolic"] = train_autoencoder(
            LorentzPatchAutoencoder, train_data, args_cli.patch,
            args_cli.latent, args_cli.ae_steps)

    helm_arms = [("helm-calm", "HELM-CALM (hyperbolic backbone)", "euclidean"),
                 ("hyperbolic", "HELM-CALM (+ hyperbolic latent)", "hyperbolic")]
    for key, label, latent_kind in helm_arms:
        if key not in arms:
            continue
        autoencoder = autoencoders[latent_kind]
        model = train_calm(args, autoencoder, train_batches, args_cli.steps,
                           args_cli.lr)

        def draw(tokens, n, model=model):
            samples, targets = model.sample_tokens(tokens, n_samples=n)
            rows = tokens.size(0)
            return samples.reshape(n, rows, -1), targets.reshape(rows, -1)

        def sampler(tokens, draw=draw):
            first, targets = draw(tokens, 1)
            second, _ = draw(tokens, 1)
            return first[0], second[0], targets

        orders, aggregate = score_brierlm(sampler, valid_batches)
        results[label] = {
            "bpb": None,
            "ae_ceiling": autoencoder_ceiling(autoencoder, valid_data),
            "top1": top1_accuracy(draw, valid_batches),
            "brier_orders": orders,
            "brierlm": aggregate,
        }

    if "euclidean-calm" in arms:
        autoencoder = autoencoders["euclidean"]
        model = train_euclidean_calm(args, autoencoder, train_batches,
                                     args_cli.steps, args_cli.lr, args_cli.patch)

        def draw(tokens, n, model=model, ae=autoencoder):
            return euclidean_calm_draw(model, ae, tokens, n)

        def sampler(tokens, draw=draw):
            first, targets = draw(tokens, 1)
            second, _ = draw(tokens, 1)
            return first[0], second[0], targets

        orders, aggregate = score_brierlm(sampler, valid_batches)
        results["CALM (Euclidean backbone, control)"] = {
            "bpb": None,
            "ae_ceiling": autoencoder_ceiling(autoencoder, valid_data),
            "top1": top1_accuracy(draw, valid_batches),
            "brier_orders": orders,
            "brierlm": aggregate,
        }

    print(f"{'model':36s} {'top-1':>8s} {'BPB valid':>10s} {'BPB train':>10s} "
          f"{'brier_1':>9s} {'brier_2':>9s} {'brier_3':>9s} {'brier_4':>9s} "
          f"{'AE ceil':>9s}")
    for name, row in results.items():
        bpb = f"{row['bpb']:.4f}" if row.get("bpb") is not None else "n/a"
        bpb_train = (f"{row['bpb_train']:.4f}" if row.get("bpb_train") is not None
                     else "n/a")
        ceiling = (f"{row['ae_ceiling']:.2%}" if "ae_ceiling" in row else "n/a")
        orders = " ".join(f"{v:9.5f}" for v in row["brier_orders"])
        print(f"{name:36s} {row['top1']:8.2%} {bpb:>10s} {bpb_train:>10s} "
              f"{orders} {ceiling:>9s}")
    print("\nBPB is n/a for HELM-CALM by construction: an implicit sampler has "
          "no density.\nA uniform byte model scores BPB 8.0000 and brier_n "
          "about 1/256 = 0.0039.\n"
          "\nRead the per-order columns, not BrierLM: one zero factor collapses "
          "the geometric\nmean to 0.0000 for every arm alike (see "
          "score_brierlm). Reading brier_n:\n"
          "  ~0    diffuse and wrong -- draws neither match the target nor each "
          "other\n"
          "  < 0   mode collapse -- the two independent draws agree with each "
          "other but not\n"
          "        with the target; -1 is total collapse, which is what the "
          "collision term\n"
          "        exists to punish\n"
          "  > 0   real signal; this is the only region where a comparison "
          "between arms means\n"
          "        anything")


if __name__ == "__main__":
    main()
