"""Does a hyperbolic latent close the 2.94-point gap?

``DID_IT_WORK.md`` §5 measured, over five seeds, that HELM-CALM trails a
Euclidean control by 2.94 points (4.6 standard errors) with twelve times the
seed variance, and located the one remaining Euclidean seam: the autoencoder's
latent. ``CALM/hyperbolic_latent.py`` closes it -- wrapped-normal posterior on
the hyperboloid, geodesic energy score, and a head that no longer has to step
off the manifold at its final layer.

This runs the same protocol as ``seed_variance.py`` with a third arm added, so
the numbers are directly comparable to the table already in the docs.

**Held-out evaluation.** ``seed_variance.py``, whose table reached
``DID_IT_WORK.md`` §5, trained and evaluated on the *same* sixteen batches. Its
numbers are therefore training-set accuracy over 1024 tokens seen ~250 times
each -- a memorisation measure, not a generalisation one. This script draws
evaluation sequences from the same grammar under disjoint seeds and reports both,
so the difference between memorising and generalising is visible rather than
assumed. See ``EVALUATION.md``.

**The confound this controls for.** The two arms use *different autoencoders*,
and downstream accuracy cannot exceed what the autoencoder can reconstruct. A
hyperbolic latent that simply reconstructs worse would look like a loss even if
the generative head improved. So each arm's autoencoder ceiling -- teacher-forced
round-trip accuracy on the same data -- is measured and reported alongside, and
the headline comparison is accuracy *as a fraction of that ceiling*.

Run: ``python CALM/experiments/hyperbolic_latent_seeds.py``
"""

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments"):
    sys.path.insert(0, str(extra))

import torch  # noqa: E402

from helm_calm import HelmCALM, PatchAutoencoder  # noqa: E402
from hyperbolic_latent import LorentzPatchAutoencoder  # noqa: E402
from stage1_energy_head import CalmHead, EuclideanBackbone, energy_score  # noqa: E402
from test_hierarchy import HierarchicalLanguage  # noqa: E402
from tests._config import tiny_args  # noqa: E402

STEPS, LR, SEEDS = 4000, 1e-3, [0, 1, 2, 3, 4]

LANGUAGE = HierarchicalLanguage(3, 4)
ARGS = tiny_args(vocab_size=LANGUAGE.vocab_size, dim=33, n_layers=3,
                 max_seq_len=32, original_seq_len=32)
BATCHES = [LANGUAGE.sample(2, 32, seed=i) for i in range(16)]
#: Same grammar, disjoint seeds -- sequences the model never trains on.
HELD_OUT = [LANGUAGE.sample(2, 32, seed=1000 + i) for i in range(16)]
FLAT = torch.cat([b.reshape(-1) for b in BATCHES]).view(-1, 1)
FLAT_HELD_OUT = torch.cat([b.reshape(-1) for b in HELD_OUT]).view(-1, 1)


def train_autoencoder(cls, seed=0, steps=800):
    """Same budget and data for either autoencoder class."""
    torch.manual_seed(seed)
    autoencoder = cls(LANGUAGE.vocab_size, hidden=128, latent_size=32, patch_size=1)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        rows = torch.randint(0, FLAT.size(0), (128,), generator=generator)
        loss, _ = autoencoder.elbo(FLAT[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return autoencoder.freeze()


@torch.no_grad()
def autoencoder_ceiling(autoencoder, ids=None):
    """Round-trip accuracy: the most any head built on this latent can score."""
    ids = FLAT_HELD_OUT if ids is None else ids
    posterior = autoencoder.encode(ids)
    latent = (posterior.mean if getattr(autoencoder, "is_hyperbolic", False)
              else posterior[0])
    predicted = autoencoder.decode(latent).argmax(-1)
    return (predicted == ids).float().mean().item()


def run_helm(autoencoder, seed):
    torch.manual_seed(seed)
    model = HelmCALM(ARGS, autoencoder, num_samples=8, head_kind="lorentz")
    groups = model.parameter_groups()
    params = groups["euclidean"] + groups["manifold"]
    optimizer = torch.optim.AdamW(params, lr=LR)
    model.train()
    for step in range(STEPS):
        loss = model.loss(BATCHES[step % 16])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        model.retract_manifold_parameters()
    model.eval()

    def accuracy(batches):
        correct = total = 0
        with torch.no_grad():
            for tokens in batches:
                predicted, targets = model.predict_tokens(tokens, n_samples=32)
                correct += (predicted == targets).sum().item()
                total += targets.numel()
        return correct / total

    return accuracy(BATCHES), accuracy(HELD_OUT)


def run_euclidean(autoencoder, seed, width=33):
    """The control: a Euclidean backbone and CALM's own head, unchanged."""
    torch.manual_seed(seed)
    backbone = EuclideanBackbone(ARGS.vocab_size, width, ARGS.n_layers,
                                 ARGS.n_heads, ARGS.inter_dim)
    head = CalmHead(width, autoencoder.latent_size)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)
    backbone.train()
    head.train()
    for step in range(STEPS):
        tokens = BATCHES[step % 16]
        targets = tokens[:, 1:].reshape(-1)
        with torch.no_grad():
            mean, log_std = autoencoder.encode(targets.unsqueeze(-1))
        hidden = backbone(tokens)[:, :-1].reshape(-1, width)
        samples = head.sample(hidden.unsqueeze(0).expand(8, -1, -1))
        loss = -energy_score(samples, mean, log_std).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    backbone.eval()
    head.eval()

    def accuracy(batches):
        correct = total = 0
        with torch.no_grad():
            for tokens in batches:
                targets = tokens[:, 1:].reshape(-1)
                hidden = backbone(tokens)[:, :-1].reshape(-1, width)
                drawn = head.sample(hidden.unsqueeze(0).expand(32, -1, -1))
                decoded = autoencoder.decode(drawn).argmax(-1).squeeze(-1)
                correct += (torch.mode(decoded, dim=0).values == targets).sum().item()
                total += targets.numel()
        return correct / total

    return accuracy(BATCHES), accuracy(HELD_OUT)


def summarise(name, results, ceiling):
    """``results`` is a list of ``(train, held_out)`` pairs, one per seed."""
    train = [t for t, _ in results]
    held = [h for _, h in results]
    mean, sd = statistics.mean(held), statistics.stdev(held)
    cells = " ".join(f"{v:6.2%}" for v in held)
    print(f"{name:34s} {cells} {mean:8.2%}{sd:7.2%}  {mean / ceiling:7.2%}"
          f"  {statistics.mean(train):7.2%}")
    return mean, sd


def main():
    euclidean_ae = train_autoencoder(PatchAutoencoder)
    hyperbolic_ae = train_autoencoder(LorentzPatchAutoencoder)
    ceilings = {"euclidean": autoencoder_ceiling(euclidean_ae),
                "hyperbolic": autoencoder_ceiling(hyperbolic_ae)}
    print(f"{len(SEEDS)} seeds, tree language, K=1, {STEPS} steps")
    print(f"autoencoder ceilings (held out): Euclidean {ceilings['euclidean']:.2%}, "
          f"hyperbolic {ceilings['hyperbolic']:.2%}")
    print(f"autoencoder ceilings (train):    Euclidean "
          f"{autoencoder_ceiling(euclidean_ae, FLAT):.2%}, hyperbolic "
          f"{autoencoder_ceiling(hyperbolic_ae, FLAT):.2%}\n")
    header = " ".join(f"s{s}".rjust(6) for s in SEEDS)
    print("held-out accuracy per seed; 'train' is the same models on the data "
          "they were fitted to")
    print(f"{'':34s} {header} {'mean':>8s}{'sd':>7s}  {'of ceil':>7s}  "
          f"{'train':>7s}")

    control = summarise("CALM + Euclidean (control)",
                        [run_euclidean(euclidean_ae, s) for s in SEEDS],
                        ceilings["euclidean"])
    flat = summarise("CALM + HELM, Euclidean latent",
                     [run_helm(euclidean_ae, s) for s in SEEDS],
                     ceilings["euclidean"])
    curved = summarise("CALM + HELM, hyperbolic latent",
                       [run_helm(hyperbolic_ae, s) for s in SEEDS],
                       ceilings["hyperbolic"])

    def gap(label, arm, ceiling):
        difference = control[0] / ceilings["euclidean"] - arm[0] / ceiling
        error = (control[1] ** 2 / len(SEEDS) + arm[1] ** 2 / len(SEEDS)) ** 0.5
        print(f"{label:34s} {difference:+.2%} of ceiling, "
              f"se {error:.2%}, ratio {abs(difference) / error if error else 0:.1f}")

    print()
    gap("gap, Euclidean latent", flat, ceilings["euclidean"])
    gap("gap, hyperbolic latent", curved, ceilings["hyperbolic"])
    print(f"\nseed sd: control {control[1]:.2%}, Euclidean latent {flat[1]:.2%}, "
          f"hyperbolic latent {curved[1]:.2%}")


if __name__ == "__main__":
    main()
