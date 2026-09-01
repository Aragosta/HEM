"""Integration tests for the assembled HELM-CALM.

Checks that the pieces are actually wired together — every component receives
gradient, the manifold constraint survives training, the loss falls, samples
decode to the right shapes — and that pulling CALM in has not disturbed the
optimized HELM it is built on.

Run with ``python -m pytest CALM/experiments/test_helm_calm.py -v``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CALM"))

from helm_calm import (CalmEnergyHead, HelmCALM, LorentzPatchEmbedding,  # noqa: E402
                       PatchAutoencoder, energy_score)
from helm.hypercore.manifolds import Lorentz  # noqa: E402
from tests._config import tiny_args  # noqa: E402

PATCH = 2
LATENT = 32


def warm_up(model, tokens, steps=3, lr=1e-3):
    """Take a few optimizer steps so the head leaves its zero initialisation.

    CALM zero-initialises the generative head's final projection, so
    ``d(output)/d(hidden) = 0`` and the **backbone receives exactly no gradient
    on step 0**. Measured: backbone gradient 0.000e+00 at step 0, 7.3e-06 at
    step 1. That is by design -- the head starts neutral -- but it means any
    gradient-flow check performed at initialisation will conclude, wrongly, that
    the model is disconnected.
    """
    groups = model.parameter_groups()
    params = groups["euclidean"] + groups["manifold"]
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    for _ in range(steps):
        model.zero_grad()
        model.loss(tokens).backward()
        optimizer.step()
        model.retract_manifold_parameters()
    return model


@pytest.fixture(scope="module")
def pieces():
    """A trained-enough autoencoder and a model built on it."""
    torch.manual_seed(0)
    args = tiny_args()
    autoencoder = PatchAutoencoder(args.vocab_size, hidden=128,
                                   latent_size=LATENT, patch_size=PATCH)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(400):
        ids = torch.randint(0, args.vocab_size, (128, PATCH), generator=generator)
        loss, _ = autoencoder.elbo(ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model = HelmCALM(args, autoencoder, num_samples=4)
    return args, model


def batch(args, rows=2, length=24, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, args.vocab_size, (rows, length), generator=generator)


# ------------------------------------------------------------------- assembly

def test_backbone_has_no_vocabulary_head(pieces):
    """CALM's whole point: the 128256-wide projection is gone."""
    _, model = pieces
    assert not hasattr(model.backbone, "head")


def test_optimized_helm_is_still_optimized(pieces):
    """The backbone must keep every optimization; CALM changes the head, not it."""
    from helm.modules.lorentz_ops import LorentzResidual
    from helm.modules.mice import LorentzMoE, LorentzSwiGLU

    _, model = pieces
    block = model.backbone.layers[0]
    assert block.attn.attn_impl == "flash"
    assert isinstance(block.attn_res, LorentzResidual)
    moe = next(layer.ffn for layer in model.backbone.layers
               if isinstance(layer.ffn, LorentzMoE))
    assert isinstance(moe.experts[0], LorentzSwiGLU), "fused experts expected"


def test_autoencoder_is_frozen(pieces):
    _, model = pieces
    assert not any(p.requires_grad for p in model.autoencoder.parameters())
    assert not model.autoencoder.training


def test_shapes_flow_through(pieces):
    args, model = pieces
    tokens = batch(args)
    hidden = model.hidden_states(tokens)
    assert hidden.shape == (tokens.size(0), tokens.size(1) // PATCH, args.dim)
    samples, targets = model.sample_tokens(tokens, n_samples=3)
    positions = tokens.size(0) * (tokens.size(1) // PATCH - 1)
    assert samples.shape == (3, positions, PATCH)
    assert targets.shape == (positions, PATCH)


def test_patch_embedding_output_is_on_the_manifold(pieces):
    """The reason for the custom patch embedding: activations must stay on it."""
    args, model = pieces
    tokens = batch(args)
    embedded = model.backbone.embed(tokens)
    patched = model.patch_embed(embedded)
    squared = patched ** 2
    quad = -squared[..., :1] + squared[..., 1:].sum(-1, keepdim=True)
    torch.testing.assert_close(quad, -model.backbone.manifold_hidden.c.expand_as(quad),
                               rtol=1e-4, atol=1e-4)


def test_rejects_sequence_not_divisible_by_patch(pieces):
    args, model = pieces
    with pytest.raises(ValueError, match="not a multiple of patch size"):
        model.hidden_states(batch(args, length=23))


# ------------------------------------------------------------------ gradients

def test_every_component_receives_gradient(pieces):
    """A dead component is the classic silent failure in an assembled model.

    Two families of parameter legitimately get no gradient on a given step, and
    both are inherited from HELM rather than introduced by CALM:

    * ``LorentzEmbeddings.add_pos`` -- an LResNet used only when
      ``posit_embed=True``. HELM runs ``posit_embed=False``, so it is dead weight
      at any batch size.
    * unrouted MoE experts -- the router sends no token to them this step.

    Everything else must receive one. (These two are also why
    ``find_unused_parameters`` has to stay True under DDP.)
    """
    args, model = pieces
    warm_up(model, batch(args))
    model.zero_grad()
    model.loss(batch(args)).backward()

    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    unexplained = [name for name in missing
                   if "add_pos" not in name and ".experts." not in name]
    assert not unexplained, f"no gradient reached: {unexplained}"

    # And each major component must have at least one non-zero gradient, not
    # merely a zero-filled tensor.
    for component in ("backbone.embed", "backbone.layers", "patch_embed", "head"):
        grads = [p.grad.abs().max().item()
                 for name, p in model.named_parameters()
                 if name.startswith(component) and p.grad is not None]
        assert grads and max(grads) > 0, f"{component} received only zero gradients"


def test_routing_starts_collapsed_and_balancing_recovers_it(pieces):
    """HELM's router is degenerate at initialisation; the bias update fixes it.

    At init every token picks the same top-k experts, so half of them receive no
    tokens at all. This is HELM's own behaviour -- the upstream and optimized
    models both do it -- not something the patch embedding or the energy head
    introduces.

    The auxiliary-loss-free bias update (DeepSeek-V3 section 2.1.2) is what
    recovers it, driving utilisation to uniform within a few hundred updates.
    **Upstream ships that update commented out**, so a released HELM run trains
    with permanently collapsed routing; re-enabling it is one of this port's
    fixes (``--balance_update``). This test guards that fix.
    """
    args, model = pieces
    from helm.modules.mice import LorentzMoE

    moes = [layer.ffn for layer in model.backbone.layers
            if isinstance(layer.ffn, LorentzMoE)]
    assert moes, "expected at least one MiCE layer"
    tokens = batch(args, rows=8)

    def utilisation():
        inputs, _ = model._aligned(tokens)
        hidden = model.hidden_states(inputs).reshape(-1, args.dim)
        counts = []
        for moe in moes:
            _, indices, _ = moe.gate(hidden)
            counts.append(torch.bincount(indices.reshape(-1),
                                         minlength=moe.n_routed_experts))
        return counts

    with torch.no_grad():
        before = utilisation()
    assert min((c > 0).sum().item() for c in before) < moes[0].n_routed_experts, \
        "expected collapsed routing at initialisation"

    with torch.no_grad():
        for _ in range(800):
            inputs, _ = model._aligned(tokens)
            hidden = model.hidden_states(inputs).reshape(-1, args.dim)
            for moe in moes:
                _, indices, _ = moe.gate(hidden)
                moe.gate.update_bias(indices)
        after = utilisation()

    for index, counts in enumerate(after):
        assert (counts > 0).all(), \
            f"MiCE layer {index} still starving experts after balancing: {counts.tolist()}"


def test_no_gradient_reaches_the_frozen_autoencoder(pieces):
    args, model = pieces
    model.zero_grad()
    model.loss(batch(args)).backward()
    assert all(p.grad is None for p in model.autoencoder.parameters())


def test_parameter_groups_partition_the_model(pieces):
    _, model = pieces
    groups = model.parameter_groups()
    grouped = {id(p) for group in groups.values() for p in group}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert grouped == trainable, "parameter groups must cover exactly the trainable set"
    assert len(groups["manifold"]) >= 1


# ------------------------------------------------------------------- learning

def test_loss_decreases_and_manifold_holds(pieces):
    """The end-to-end check: it trains, and it stays on the manifold while doing so."""
    args, _ = pieces
    torch.manual_seed(0)
    autoencoder = PatchAutoencoder(args.vocab_size, hidden=128,
                                   latent_size=LATENT, patch_size=PATCH)
    model = HelmCALM(args, autoencoder, num_samples=4)

    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW(groups["euclidean"] + groups["manifold"], lr=1e-3)

    # A learnable stream, so "loss went down" means something.
    generator = torch.Generator().manual_seed(3)
    start = torch.randint(0, args.vocab_size, (2, 1), generator=generator)
    stride = torch.randint(1, 4, (2, 1), generator=generator)
    tokens = (start + stride * torch.arange(24).unsqueeze(0)) % args.vocab_size

    model.train()
    losses = []
    for _ in range(60):
        loss = model.loss(tokens)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            groups["euclidean"] + groups["manifold"], 1.0)
        optimizer.step()
        model.retract_manifold_parameters()
        losses.append(loss.item())

    assert losses[-1] < losses[0] - 0.5, f"loss did not fall: {losses[0]} -> {losses[-1]}"
    assert model.manifold_violation() < 1e-5, model.manifold_violation()


def test_retraction_actually_repairs_drift(pieces):
    """Without it, a Euclidean optimizer walks the embedding off the hyperboloid."""
    _, model = pieces
    with torch.no_grad():
        model.backbone.embed.embedding.add_(0.3)
    assert model.manifold_violation() > 1e-2
    model.retract_manifold_parameters()
    assert model.manifold_violation() < 1e-5


# --------------------------------------------------------------- energy score

@pytest.mark.parametrize("head_kind,final_weight", [
    ("euclidean", lambda head: head.final_layer[-1].weight),
    ("lorentz", lambda head: head.final.linear.weight),
])
def test_backbone_gradient_is_zero_at_initialisation(pieces, head_kind, final_weight):
    """Pin the zero-init behaviour, so it is not mistaken for a broken model.

    Both heads zero-initialise their final projection, as CALM does, so on step 0
    the backbone's gradient is exactly zero.
    """
    args, _ = pieces
    torch.manual_seed(0)
    autoencoder = PatchAutoencoder(args.vocab_size, hidden=128,
                                   latent_size=LATENT, patch_size=PATCH)
    fresh = HelmCALM(args, autoencoder, num_samples=4, head_kind=head_kind)
    fresh.train()
    fresh.zero_grad()
    fresh.loss(batch(args)).backward()

    # The head's own final projection does get a gradient; nothing before it does.
    assert final_weight(fresh.head).grad.abs().max() > 0
    assert fresh.backbone.embed.embedding.grad.abs().max() == 0

    warm_up(fresh, batch(args))
    fresh.zero_grad()
    fresh.loss(batch(args)).backward()
    assert fresh.backbone.embed.embedding.grad.abs().max() > 0, \
        "gradient should reach the backbone once the head has moved"


def test_lorentz_head_keeps_activations_on_the_manifold(pieces):
    """The point of LorentzEnergyHead: its internals do not leave the manifold.

    CALM's Euclidean head opens with nn.LayerNorm over all `dim` coordinates,
    including the Lorentz time coordinate -- which is structurally >= sqrt(c) and
    never negative. That drags the point off the hyperboloid before any learned
    layer runs. Worth ~7.5 accuracy points; see ../DID_IT_WORK.md.
    """
    from helm_calm import LorentzEnergyHead

    args, _ = pieces
    torch.manual_seed(0)
    autoencoder = PatchAutoencoder(args.vocab_size, hidden=128,
                                   latent_size=LATENT, patch_size=PATCH)
    model = HelmCALM(args, autoencoder, num_samples=4, head_kind="lorentz")
    assert isinstance(model.head, LorentzEnergyHead)
    assert model.input_map == "direct", "a hyperbolic head takes the point as-is"

    hidden = model.hidden_states(batch(args))
    manifold = model.backbone.manifold_out
    block = model.head.blocks[0]
    with torch.no_grad():
        y = model.head.hidden_embd(hidden)
        noise_like = model.head.noise_embd(
            torch.cat([torch.ones_like(y[..., :1]) * manifold.c.sqrt(),
                       torch.zeros(*y.shape[:-1], model.head.noise_size)], dim=-1))
        out = block(noise_like, y)

    for name, tensor in (("hidden_embd", y), ("block output", out)):
        squared = tensor ** 2
        quad = -squared[..., :1] + squared[..., 1:].sum(-1, keepdim=True)
        torch.testing.assert_close(
            quad, -manifold.c.expand_as(quad), rtol=1e-3, atol=1e-3,
            msg=lambda m, n=name: f"{n} left the manifold: {m}")


def test_energy_score_rejects_unsafe_beta():
    """beta < 1 gives NaN gradients on the self-distance term; refuse 0 and 2."""
    samples = torch.randn(4, 6, LATENT)
    mean = torch.zeros(6, LATENT)
    log_std = torch.zeros(6, LATENT)
    for beta in (0.0, 2.0, -1.0):
        with pytest.raises(ValueError, match="beta"):
            energy_score(samples, mean, log_std, beta=beta)


def test_energy_score_is_minimised_at_the_target():
    """Sanity that the scoring rule points the right way."""
    torch.manual_seed(0)
    mean = torch.zeros(1, LATENT)
    log_std = torch.full((1, LATENT), -0.5)
    losses = []
    for offset in (0.0, 0.5, 1.0, 2.0):
        samples = mean + offset + torch.randn(64, 1, LATENT) * log_std.exp()
        losses.append(-energy_score(samples, mean, log_std, n_target=500).mean().item())
    assert losses == sorted(losses), losses


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
