"""The "why" probes: mechanisms, not scores.

Accuracy says *what*. Each probe here is tied to a specific mechanism that has
already produced a wrong conclusion in this project, and exists so the same
mistake is caught by instrumentation rather than by luck:

* three results turned out to be numerical failures wearing the costume of an
  architectural finding, so :func:`numerics` runs on every cell;
* a hyperbolic latent scored 2.29% because it was pinned against a clamp at a
  third of the radius the task needed, so :func:`radius_profile` reports the
  radius *and* the pinned fraction, and the float32 cliff is a named constant;
* ``HIERARCHY.md`` asks whether patching destroys the hierarchy HELM exists to
  model, and no experiment so far has been able to answer it. :func:`delta_hyperbolicity`
  measured at token level and again at patch level is that question, made
  numerical.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

#: Tangent radius past which a Lorentz point stops being representable in float32.
#: Measured on this hardware: constraint error 0.0 at radius 4, 0.0039 at 6,
#: **0.25 at 8**, and 1.0 at 10 -- as large as the constraint itself. Any
#: hyperbolic number measured above ~6 should be doubted before it is
#: interpreted.
FLOAT32_RADIUS_CLIFF = 8.0
SAFE_RADIUS = 6.0


@torch.no_grad()
def numerics(tensor: torch.Tensor) -> Dict[str, float]:
    """Finiteness and dynamic range. Run first; a failure here voids the row."""
    finite = torch.isfinite(tensor)
    return {"nonfinite_fraction": (~finite).float().mean().item(),
            "abs_max": tensor[finite].abs().max().item() if finite.any() else float("nan")}


@torch.no_grad()
def radius_profile(points: torch.Tensor, manifold=None,
                   clamp: Optional[float] = None) -> Dict[str, float]:
    """Where activations sit relative to what float32 can represent.

    Args:
        points: Lorentz vectors ``(..., d + 1)`` when ``manifold`` is given,
            otherwise ordinary vectors whose norm is reported instead.
        clamp: the limit actually in force, if any, so ``pinned`` is measured
            against the real bound rather than a remembered one.
    """
    if manifold is not None:
        radius = manifold.logmap0(points)[..., 1:].norm(dim=-1)
        squared = points ** 2
        quad = -squared[..., :1] + squared[..., 1:].sum(-1, keepdim=True)
        violation = (quad + manifold.c).abs().max().item()
    else:
        radius = points.norm(dim=-1)
        violation = 0.0
    pinned = (0.0 if clamp is None
              else (radius > clamp - 1e-3).float().mean().item())
    return {"radius_mean": radius.mean().item(),
            "radius_max": radius.max().item(),
            "beyond_safe": (radius > SAFE_RADIUS).float().mean().item(),
            "beyond_cliff": (radius > FLOAT32_RADIUS_CLIFF).float().mean().item(),
            "pinned": pinned,
            "constraint_violation": violation}


@torch.no_grad()
def pairwise_distances(points: torch.Tensor, manifold=None,
                       limit: int = 512) -> torch.Tensor:
    """Distance matrix in the geometry the points actually live in.

    Using Euclidean distance on hyperbolic points, or the reverse, would make
    the delta below meaningless, so the metric follows the representation.
    """
    x = points.reshape(-1, points.shape[-1])[:limit].double()
    if manifold is None:
        return torch.cdist(x, x)
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    quad = (diff[..., 1:].square().sum(-1) - diff[..., 0].square()).clamp_min(0)
    sqrt_c = torch.as_tensor(manifold.c, dtype=x.dtype).sqrt()
    return 2 * sqrt_c * torch.asinh(quad.sqrt() / (2 * sqrt_c))


@torch.no_grad()
def delta_hyperbolicity(distances: torch.Tensor, samples: int = 20000,
                        seed: int = 0) -> float:
    """Gromov's four-point condition, normalised by diameter. Lower is more tree-like.

    **This is HELM's thesis as a measurable quantity.** If the hyperbolic
    backbone is not producing more tree-like representations than the Euclidean
    one, the geometry is not doing what it is claimed to do, whatever the
    accuracy column says -- and a quality difference would then need a different
    explanation.

    Normalised by the diameter so hyperbolic and Euclidean representations,
    which live at different scales, are comparable.
    """
    n = distances.shape[0]
    if n < 4:
        return float("nan")
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (samples, 4), generator=generator)
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    s1 = distances[a, b] + distances[c, d]
    s2 = distances[a, c] + distances[b, d]
    s3 = distances[a, d] + distances[b, c]
    top = torch.stack([s1, s2, s3], dim=-1).sort(dim=-1, descending=True).values
    diameter = distances.max().clamp_min(1e-12)
    return ((top[:, 0] - top[:, 1]) / (2 * diameter)).mean().item()


@torch.no_grad()
def hierarchy_flattening(token_points: torch.Tensor, patch_points: torch.Tensor,
                         manifold=None, seed: int = 0) -> Dict[str, float]:
    """Does patching flatten the structure? ``delta`` at token level vs patch level.

    The direct form of the ``HIERARCHY.md`` worry. If patch representations are
    markedly *less* tree-like than the token representations they were built
    from, patching is destroying the hierarchy HELM exists to model, and a
    negative interaction in the 2x2 has a mechanism behind it rather than being
    a bare number.

    ``ratio`` above 1 means flattening; below 1 means patching has *concentrated*
    the hierarchy, which would be the opposite of the worry and worth checking
    twice before believing.
    """
    token_delta = delta_hyperbolicity(
        pairwise_distances(token_points, manifold), seed=seed)
    patch_delta = delta_hyperbolicity(
        pairwise_distances(patch_points, manifold), seed=seed)
    ratio = (patch_delta / token_delta) if token_delta > 1e-9 else float("nan")
    return {"token_delta": token_delta, "patch_delta": patch_delta,
            "flattening_ratio": ratio}


@torch.no_grad()
def collapse(samples: torch.Tensor) -> Dict[str, float]:
    """Is the model emitting one thing regardless of the noise it is given?

    ``agreement`` is the fraction of positions where two independent draws
    coincide. Near 1 with poor accuracy is mode collapse -- which is what a
    negative BrierLM means, and why that metric's sign has to be read rather
    than its magnitude alone. ``distinct`` is the share of the vocabulary the
    model ever emits.
    """
    if samples.shape[0] < 2:
        return {"agreement": float("nan"), "distinct": float("nan")}
    agreement = (samples[0] == samples[1]).float().mean().item()
    distinct = samples.unique().numel() / max(samples.max().item() + 1, 1)
    return {"agreement": agreement, "distinct": distinct}
