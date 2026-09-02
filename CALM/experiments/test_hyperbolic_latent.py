"""Tests for the hyperbolic latent: geometry, density, and the estimator.

The geometric pieces here are easy to write plausibly and get subtly wrong, so
each is checked against something independent rather than against itself:
distance against ``geoopt``, the density against numerical integration, and the
energy score against the property that makes it worth using -- that its optimum
is the true distribution.

Run with ``python -m pytest CALM/experiments/test_hyperbolic_latent.py -v``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "CALM"):
    sys.path.insert(0, str(extra))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from hyperbolic_latent import (LorentzPatchAutoencoder, WrappedNormal,  # noqa: E402
                               lorentz_distance, lorentz_energy_score)
from helm_calm import HelmCALM, PatchAutoencoder  # noqa: E402
from tests._config import tiny_args  # noqa: E402


@pytest.fixture(scope="module")
def manifold():
    return Lorentz(1.0)


def on_manifold(x, c=1.0):
    quad = -x[..., 0] ** 2 + x[..., 1:].square().sum(-1)
    return (quad + c).abs().max().item()


# ------------------------------------------------------------------- distance

def test_distance_matches_geoopt(manifold):
    """Our stable form must agree with the reference to double precision."""
    torch.manual_seed(0)
    x = manifold.random_normal((64, 8), std=0.8).double()
    y = manifold.random_normal((64, 8), std=0.8).double()
    assert torch.allclose(lorentz_distance(x, y), manifold.dist(x, y), atol=1e-10)


def test_distance_beats_arccosh_near_zero(manifold):
    """The reason for the asinh form: arccosh(1 + eps) loses half the mantissa.

    Two points a hair apart. The textbook expression evaluates arccosh at
    1 + 1e-14 in float32, where the cancellation has already destroyed the
    answer; the difference form does not.
    """
    x = manifold.random_normal((256, 8), std=0.5)
    y = manifold.projx(x + torch.randn_like(x) * 1e-4)
    exact = lorentz_distance(x.double(), y.double())

    ours = lorentz_distance(x, y).double()
    inner = -(-x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(-1, True))
    textbook = torch.acosh(inner.clamp_min(1.0)).squeeze(-1).double()

    ours_error = (ours - exact).abs().max()
    textbook_error = (textbook - exact).abs().max()
    assert ours_error < textbook_error
    assert ours_error < 1e-6


def test_distance_gradient_finite_at_coincidence(manifold):
    """Self-distances appear on the pairwise term's diagonal; they must not NaN."""
    x = manifold.random_normal((16, 6), std=0.5).requires_grad_(True)
    grad = torch.autograd.grad(lorentz_distance(x, x).sum(), x)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().max() == 0


def test_distance_is_a_metric(manifold):
    """Symmetry, identity, and the triangle inequality, sampled."""
    torch.manual_seed(1)
    x, y, z = (manifold.random_normal((200, 6), std=0.7).double() for _ in range(3))
    assert torch.allclose(lorentz_distance(x, y), lorentz_distance(y, x))
    assert lorentz_distance(x, x).abs().max() < 1e-12
    assert (lorentz_distance(x, z) <= lorentz_distance(x, y)
            + lorentz_distance(y, z) + 1e-9).all()


def test_distance_scales_with_curvature():
    """A hyperboloid of scale c has distances sqrt(c) times the unit one."""
    unit, wide = Lorentz(1.0), Lorentz(4.0)
    torch.manual_seed(2)

    def exact(size):
        # random_normal is built in float32, so promoting it to double leaves a
        # ~1e-7 constraint residual that doubles under the scaling below.
        # Recomputing the time coordinate in double removes it.
        space = unit.random_normal(size, std=0.6).double()[..., 1:]
        time = (space.square().sum(-1, keepdim=True) + 1.0).sqrt()
        return torch.cat([time, space], dim=-1)

    x, y = exact((32, 5)), exact((32, 5))
    scaled_x, scaled_y = x * 2.0, y * 2.0  # the c=4 hyperboloid
    assert on_manifold(scaled_x, c=4.0) < 1e-9
    assert torch.allclose(lorentz_distance(scaled_x, scaled_y, c=wide.c.double()),
                          2.0 * lorentz_distance(x, y), atol=1e-9)


# ------------------------------------------------------------- wrapped normal

def test_samples_lie_on_the_manifold(manifold):
    torch.manual_seed(3)
    mean = manifold.random_normal((32, 9), std=0.5)
    posterior = WrappedNormal(manifold, mean, torch.full((32, 8), -1.0))
    assert on_manifold(posterior.sample(16)) < 1e-5


def test_sampling_is_reparameterised(manifold):
    """Gradient must reach the mean through the sample, or the ELBO is broken."""
    tangent = torch.zeros(4, 8, requires_grad=True)
    mean = manifold.expmap0(torch.cat([torch.zeros(4, 1), tangent], dim=-1))
    log_std = torch.full((4, 8), -1.0, requires_grad=True)
    z, _ = WrappedNormal(manifold, mean, log_std).rsample()
    grad_mean, grad_std = torch.autograd.grad(z.sum(), [tangent, log_std])
    assert torch.isfinite(grad_mean).all() and grad_mean.abs().sum() > 0
    assert torch.isfinite(grad_std).all() and grad_std.abs().sum() > 0


def test_density_integrates_to_one(manifold):
    """The Jacobian correction is the whole content of log_prob; check it holds.

    Integrating over H^2 in polar tangent coordinates at the mean: the measure is
    ``sinh(r) dr dtheta``, and the wrapped density expressed in the tangent
    variable already carries its own ``(sinh r / r)^(d-1)`` factor, so the
    integral of ``exp(log_prob(v))`` against ``dv`` over R^2 must be 1.
    """
    dim = 2
    mean = manifold.origin(dim + 1).double()
    posterior = WrappedNormal(manifold, mean, torch.full((dim,), 0.0).double())
    grid = torch.linspace(-8, 8, 601, dtype=torch.float64)
    vx, vy = torch.meshgrid(grid, grid, indexing="ij")
    v = torch.stack([vx, vy], dim=-1)
    density = posterior.log_prob(v).exp()
    cell = (grid[1] - grid[0]) ** 2
    # exp(log_prob) is the density w.r.t. the manifold measure; converting to dv
    # multiplies back by the same Jacobian, giving a plain Gaussian integral.
    radius = v.square().sum(-1).sqrt().clamp_min(1e-12)
    jacobian = (torch.sinh(radius) / radius) ** (dim - 1)
    total = (density * jacobian * cell).sum()
    assert abs(total.item() - 1.0) < 1e-3


def test_density_is_the_gaussian_when_the_correction_vanishes(manifold):
    """For d = 1 the Jacobian exponent is zero and log_prob is exactly N(v;0,s)."""
    posterior = WrappedNormal(manifold, manifold.origin(2).double(),
                              torch.tensor([-0.3], dtype=torch.float64))
    v = torch.linspace(-3, 3, 11, dtype=torch.float64).unsqueeze(-1)
    expected = torch.distributions.Normal(0.0, math.exp(-0.3)).log_prob(v).squeeze(-1)
    assert torch.allclose(posterior.log_prob(v), expected, atol=1e-12)


def test_kl_is_zero_for_the_prior_itself(manifold):
    """KL(p || p) estimated over many samples must sit at zero."""
    torch.manual_seed(4)
    origin = manifold.origin(9).expand(4096, 9).contiguous()
    posterior = WrappedNormal(manifold, origin, torch.zeros(4096, 8))
    assert abs(posterior.kl_to_origin_prior().mean().item()) < 0.15


def test_kl_grows_with_displacement(manifold):
    """A posterior pushed away from the origin must cost more."""
    torch.manual_seed(5)
    values = []
    for radius in (0.0, 0.5, 1.5):
        tangent = torch.full((2048, 8), radius / math.sqrt(8))
        mean = manifold.expmap0(torch.cat([torch.zeros(2048, 1), tangent], -1))
        values.append(WrappedNormal(manifold, mean, torch.zeros(2048, 8))
                      .kl_to_origin_prior().mean().item())
    assert values[0] < values[1] < values[2]


# -------------------------------------------------------------- energy score

def test_energy_score_is_maximised_at_the_truth(manifold):
    """The property the objective exists for: propriety.

    Score a family of candidate distributions against a fixed target; the one
    centred on the target must win. This is what would break if the geodesic
    distance were not conditionally negative definite.
    """
    torch.manual_seed(6)
    tangent = torch.zeros(1, 6)
    target_mean = manifold.expmap0(torch.cat([torch.zeros(1, 1), tangent], -1))
    target = WrappedNormal(manifold, target_mean, torch.full((1, 6), -0.7))

    scores = []
    for offset in (0.0, 0.3, 0.9, 1.8):
        shift = torch.full((1, 6), offset / math.sqrt(6))
        mean = manifold.expmap0(torch.cat([torch.zeros(1, 1), shift], -1))
        samples = WrappedNormal(manifold, mean, torch.full((1, 6), -0.7)).sample(2048)
        scores.append(lorentz_energy_score(samples, target, n_target=2048).item())
    assert scores == sorted(scores, reverse=True), scores


def test_energy_score_prefers_the_right_spread(manifold):
    """Propriety in the second argument too: over- and under-dispersion both cost."""
    torch.manual_seed(7)
    mean = manifold.origin(7).expand(1, 7).contiguous()
    target = WrappedNormal(manifold, mean, torch.full((1, 6), -0.7))
    scores = {}
    for log_std in (-2.0, -0.7, 0.5):
        samples = WrappedNormal(manifold, mean, torch.full((1, 6), log_std)).sample(2048)
        scores[log_std] = lorentz_energy_score(samples, target, n_target=2048).item()
    assert scores[-0.7] > scores[-2.0] and scores[-0.7] > scores[0.5], scores


def test_energy_score_gradient_is_finite(manifold):
    torch.manual_seed(8)
    mean = manifold.origin(7).expand(1, 7).contiguous()
    target = WrappedNormal(manifold, mean, torch.full((1, 6), -0.7))
    tangent = torch.zeros(8, 1, 6, requires_grad=True)
    samples = manifold.expmap0(torch.cat([torch.zeros(8, 1, 1), tangent], -1))
    grad = torch.autograd.grad(lorentz_energy_score(samples, target).sum(), tangent)[0]
    assert torch.isfinite(grad).all()


# ------------------------------------------------------------- autoencoder

@pytest.fixture(scope="module")
def autoencoder():
    torch.manual_seed(0)
    model = LorentzPatchAutoencoder(97, hidden=64, latent_size=16, patch_size=2,
                                    layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 97, (256, 2), generator=generator)
    for _ in range(400):
        loss, recon = model.elbo(ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, ids, recon.item()


def test_autoencoder_reconstructs(autoencoder):
    model, ids, recon = autoencoder
    assert recon < 0.5
    with torch.no_grad():
        assert (model.decode(model.encode(ids).mean).argmax(-1) == ids).float().mean() > 0.9


def test_autoencoder_latent_is_on_the_manifold(autoencoder):
    model, ids, _ = autoencoder
    with torch.no_grad():
        assert on_manifold(model.encode(ids).sample(4)) < 1e-5


def test_model_rejects_a_euclidean_head_with_a_curved_latent(autoencoder):
    model, _, _ = autoencoder
    args = tiny_args()
    curved = LorentzPatchAutoencoder(args.vocab_size, hidden=32, latent_size=8,
                                     patch_size=2, layers=1)
    with pytest.raises(ValueError, match="head_kind"):
        HelmCALM(args, curved, head_kind="euclidean")


def test_model_rejects_beta_other_than_one():
    args = tiny_args()
    curved = LorentzPatchAutoencoder(args.vocab_size, hidden=32, latent_size=8,
                                     patch_size=2, layers=1)
    with pytest.raises(ValueError, match="beta must be 1.0"):
        HelmCALM(args, curved, beta=1.5)


def test_head_stays_on_the_manifold_end_to_end():
    """The point of the exercise: no seam left where the head leaves H^n."""
    torch.manual_seed(0)
    args = tiny_args()
    curved = LorentzPatchAutoencoder(args.vocab_size, hidden=32, latent_size=8,
                                     patch_size=2, layers=1)
    model = HelmCALM(args, curved, num_samples=4)
    assert model.hyperbolic_latent and model.head.on_manifold
    tokens = torch.randint(0, args.vocab_size, (2, 16))
    with torch.no_grad():
        hidden = model.head_input(model.hidden_states(tokens))
        samples = model.head(hidden.reshape(-1, model.head_width)
                             .unsqueeze(0).expand(4, -1, -1))
    assert samples.shape[-1] == curved.latent_size + 1
    assert on_manifold(samples) < 1e-4


def test_training_reduces_the_loss():
    torch.manual_seed(0)
    args = tiny_args()
    curved = LorentzPatchAutoencoder(args.vocab_size, hidden=64, latent_size=16,
                                     patch_size=2, layers=1)
    optimizer = torch.optim.AdamW(curved.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        loss, _ = curved.elbo(torch.randint(0, args.vocab_size, (128, 2),
                                            generator=generator))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model = HelmCALM(args, curved, num_samples=4)
    tokens = torch.randint(0, args.vocab_size, (2, 16), generator=generator)
    groups = model.parameter_groups()
    params = groups["euclidean"] + groups["manifold"]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    losses = []
    for _ in range(60):
        loss = model.loss(tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.retract_manifold_parameters()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses)
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
    assert model.manifold_violation().item() < 1e-4


# ------------------------------------------------- float32 radius budget

def test_constraint_degrades_past_radius_eight(manifold):
    """The measurement behind MAX_TANGENT_RADIUS, kept as a regression guard.

    Coordinates grow as cosh(r), so the hyperboloid constraint asks float32 to
    cancel two quantities of size e^(2r)/4 and land on -c. Past radius ~8 that
    cancellation is below the noise floor and the point is no longer on the
    manifold in any useful sense.
    """
    def error(radius):
        tangent = torch.zeros(1, 32)
        tangent[0, 0] = radius
        point = manifold.expmap0(torch.cat([torch.zeros(1, 1), tangent], -1))
        return on_manifold(point)

    assert error(4.0) < 1e-4
    assert error(6.0) < 1e-2
    assert error(10.0) > 0.5      # error as large as the constraint itself


def test_clamp_keeps_the_posterior_representable():
    """Without the clamp the wrapped normal walks out of float32 and NaNs.

    Measured before the fix: radius 9.46 after nine steps at latent width 32,
    with NaN in both the sample and the KL.
    """
    from hyperbolic_latent import MAX_TANGENT_RADIUS, WrappedNormal as WN
    tangent = torch.randn(64, 32) * 10.0
    clamped = WN.clamp_tangent(tangent)
    assert clamped.norm(dim=-1).max() <= MAX_TANGENT_RADIUS + 1e-5
    # Direction preserved, and differentiable rather than truncated.
    short = torch.randn(8, 32) * 0.01
    assert torch.allclose(WN.clamp_tangent(short), short, atol=1e-6)
    live = (torch.randn(4, 32) * 10.0).requires_grad_(True)
    WN.clamp_tangent(live).sum().backward()
    assert torch.isfinite(live.grad).all() and live.grad.abs().sum() > 0


def test_autoencoder_survives_a_wide_latent():
    """Latent width 64 NaN'd within five steps before the clamp."""
    torch.manual_seed(0)
    model = LorentzPatchAutoencoder(97, hidden=64, latent_size=64, patch_size=2,
                                    layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(60):
        ids = torch.randint(0, 97, (128, 2), generator=generator)
        loss, _ = model.elbo(ids)
        assert torch.isfinite(loss), "wrapped normal left float32"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
