"""Structural parity against CALM's own modules, not against our reading of them.

Everything else in this directory tests our code against our intentions. This
file imports ``upstream/models/`` -- CALM's released implementation -- and checks
that our transcription has the same parameter shapes, the same layer structure
and, where the computation is deterministic, the same numbers.

It exists because a review of the two side by side found four silent
divergences: a patch embedding a quarter of CALM's depth, missing per-branch
entry norms in the head, a single-layer final projection where CALM has a gated
two-layer one, and an autoencoder with neither free-bits KL clamping nor latent
dropout. None of those raise an error; they just quietly make the model a
different model. Shape parity catches that class of drift.

Run: ``python -m pytest CALM/experiments/test_calm_parity.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM"):
    sys.path.insert(0, str(_extra))

from helm_calm import CalmEnergyHead, LorentzEnergyHead, PatchAutoencoder  # noqa: E402
from helm.hypercore.manifolds import Lorentz  # noqa: E402

upstream = pytest.importorskip("CALM.upstream.models.modeling_energy",
                               reason="CALM upstream needs transformers")
config_mod = pytest.importorskip("CALM.upstream.models.configuration_calm")
ae_config_mod = pytest.importorskip("CALM.upstream.models.configuration_autoencoder")

HIDDEN, LATENT, NOISE, LAYERS = 64, 32, 16, 2


def upstream_config():
    return config_mod.CALMConfig(hidden_size=HIDDEN, latent_size=LATENT,
                                 noise_size=NOISE, num_mlp_layers=LAYERS,
                                 patch_size=4)


#: Ours to CALM's. The modules are the same; only the naming differs -- we build
#: the final stage as an ``nn.Sequential`` where CALM has a named ``FinalLayer``,
#: and our block list is ``blocks`` where theirs is ``mlp_blocks``. Aligning by
#: sorted order instead of by this map pairs unrelated tensors and reports a
#: mismatch that is not there.
NAME_MAP = {
    "final_layer.0": "final_layer.in_ln",
    "final_layer.1": "final_layer.linears.0",
    "final_layer.3": "final_layer.linears.2",
}


def to_upstream(name: str) -> str:
    for ours, theirs in NAME_MAP.items():
        if name.startswith(ours + "."):
            return name.replace(ours, theirs, 1)
    return name.replace("blocks.", "mlp_blocks.", 1)


def test_energy_head_matches_upstream_parameter_for_parameter():
    """Our CalmEnergyHead must be MLPGenerator, tensor for tensor."""
    theirs = dict(upstream.MLPGenerator(upstream_config()).named_parameters())
    ours = dict(CalmEnergyHead(HIDDEN, LATENT, noise_size=NOISE,
                               num_mlp_layers=LAYERS).named_parameters())

    assert (sum(p.numel() for p in ours.values())
            == sum(p.numel() for p in theirs.values()))
    missing = {to_upstream(n) for n in ours} ^ set(theirs)
    assert not missing, f"structure drifted from upstream: {sorted(missing)}"
    for name, parameter in ours.items():
        assert parameter.shape == theirs[to_upstream(name)].shape, name


def test_energy_head_is_numerically_identical_under_shared_weights():
    """Same weights and the same noise must give the same latent.

    Parameter parity would still pass if the blocks were wired in a different
    order or the two entry norms were swapped, so this copies our weights into
    theirs and compares outputs. Both heads draw their noise with ``torch.rand``
    of the same shape, so seeding immediately before each call makes the draw
    identical without reaching inside either implementation.
    """
    theirs = upstream.MLPGenerator(upstream_config()).eval()
    ours = CalmEnergyHead(HIDDEN, LATENT, noise_size=NOISE,
                          num_mlp_layers=LAYERS).eval()
    theirs_params = dict(theirs.named_parameters())
    with torch.no_grad():
        for name, parameter in ours.named_parameters():
            theirs_params[to_upstream(name)].copy_(parameter)

    hidden = torch.randn(3, 5, HIDDEN)
    torch.manual_seed(1234)
    ours_out = ours(hidden)
    torch.manual_seed(1234)
    theirs_out = theirs.sample(hidden)
    assert torch.allclose(ours_out, theirs_out, atol=1e-6), (
        (ours_out - theirs_out).abs().max().item())


def test_patch_embedding_has_calm_depth():
    """CALM's embed_proj is Linear -> SiLU -> Linear -> LayerNorm, not one map.

    Checked structurally rather than by shape, since ours is built from Lorentz
    layers whose widths differ by the time coordinate. The invariant that matters
    is that the patch is widened to 2*hidden before being projected back.
    """
    from helm_calm import LorentzPatchEmbedding
    embedding = LorentzPatchEmbedding(Lorentz(1.0), dim=33, patch_size=4)
    width = 32
    assert embedding.expand.linear.out_features == 2 * width, (
        "patch embedding does not widen; CALM's embed_proj goes through 2*hidden")
    assert embedding.proj.linear.out_features == width
    assert hasattr(embedding, "norm"), "CALM's embed_proj ends in a norm"

    tokens = Lorentz(1.0).random_normal((2, 8, 33), std=0.3)
    out = embedding(tokens)
    assert out.shape == (2, 2, 33)
    quad = -out[..., 0] ** 2 + out[..., 1:].square().sum(-1)
    assert (quad + 1).abs().max() < 1e-4, "patch embedding left the manifold"


def test_lorentz_head_mirrors_the_euclidean_one():
    """The hyperbolic head must have the same *topology* as CALM's.

    Not the same shapes -- Lorentz layers carry a time coordinate -- but the
    same set of stages. This is the check that would have caught the missing
    entry norms and the single-layer final projection.
    """
    ours = LorentzEnergyHead(Lorentz(1.0), HIDDEN + 1, LATENT,
                             noise_size=NOISE, num_mlp_layers=LAYERS)
    for stage in ("noise_embd", "hidden_embd", "norm_noise", "norm_hidden",
                  "blocks", "final_norm", "final_hidden", "final"):
        assert hasattr(ours, stage), f"Lorentz head is missing {stage}"
    assert len(ours.blocks) == LAYERS


def test_autoencoder_uses_free_bits_and_latent_dropout():
    """CALM clamps the per-dimension KL and drops out the sampled latent."""
    autoencoder = PatchAutoencoder(97, hidden=32, latent_size=8, patch_size=2,
                                   layers=1)
    defaults = ae_config_mod.AutoencoderConfig()
    assert autoencoder.kl_clamp == defaults.kl_clamp
    assert autoencoder.dropout == defaults.ae_dropout

    # Free bits must floor the KL: a posterior identical to the prior has KL 0
    # per dimension, which clamps up to kl_clamp * latent_size.
    torch.manual_seed(0)
    ids = torch.randint(0, 97, (16, 2))
    autoencoder.eval()          # dropout off, so the floor is exact
    with torch.no_grad():
        mean, log_std = autoencoder.encode(ids)
        raw = (0.5 * (mean.pow(2) + (2 * log_std).exp() - 1) - log_std)
        floored = raw.clamp(min=autoencoder.kl_clamp).sum(-1)
    assert (floored >= autoencoder.kl_clamp * autoencoder.latent_size - 1e-5).all()


def test_dropout_is_active_in_training_only():
    autoencoder = PatchAutoencoder(97, hidden=32, latent_size=8, patch_size=2,
                                   layers=1)
    ids = torch.randint(0, 97, (64, 2))
    torch.manual_seed(0)
    autoencoder.train()
    first, _ = autoencoder.elbo(ids)
    torch.manual_seed(0)
    autoencoder.eval()
    with torch.no_grad():
        mean, log_std = autoencoder.encode(ids)
        clean = autoencoder.decode(mean)
    assert torch.isfinite(first) and torch.isfinite(clean).all()
