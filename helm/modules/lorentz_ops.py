"""Fused Lorentz primitives.

The GEMMs in HELM-MiCE have optimized kernels behind them; the hyperbolic glue
between them does not. Profiling a training step shows ~53% of the time in
``mm``/``bmm``/``addmm`` and ~28% spread across hundreds of small ``mul``,
``pow``, ``sum``, ``div``, ``cat`` and ``copy_`` calls -- the cost of carrying a
point's time coordinate around, discarding it, and deriving it again at the next
layer.

This module fuses the two primitives that dominate that overhead. Both are exact
rewrites, verified against the HyperCore originals.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def project_to_manifold(x_space: torch.Tensor, c: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Lift a space-like vector onto the hyperboloid: ``x_0 = sqrt(|x_s|^2 + c)``."""
    x_time = (x_space.square().sum(dim=-1, keepdim=True) + c).clamp_min(eps).sqrt()
    return torch.cat([x_time, x_space], dim=-1)


class LorentzResidual(nn.Module):
    """Drop-in replacement for ``hypercore.nn.LResNet``, fused.

    Computes the Lorentzian residual

        ave  = x + w * y
        out  = sqrt(c) * ave / sqrt(|<ave, ave>_L|)      (then optionally rescaled)

    identically to the original, but in four passes over the activation instead
    of seven, by

    * never materialising the full normalised vector when a scale is applied --
      the original computes ``sqrt(c) * ave / denom`` over all ``dim`` components
      and then immediately throws the time component away, keeping only
      ``[..., 1:]`` to rescale;
    * folding ``scale``, ``sqrt(c)`` and ``1/denom`` into a single per-row
      coefficient, so one broadcast multiply replaces a multiply, a divide and a
      second multiply;
    * reusing the ``sum(ave_s^2)`` reduction. The original reduces twice: once
      for the Minkowski inner product and again for the new time coordinate. But
      the second is just ``coef^2 * sum(ave_s^2)``, because the rescaled space
      part is a scalar multiple of ``ave_s``.

    Parameters, their names and their shapes match ``LResNet`` exactly, so state
    dicts are interchangeable.

    Args:
        manifold_in: input manifold.
        weight: fixed residual weight. When ``None`` the weight is learnable.
        batch_size: give each row its own learnable weight.
        use_scale: rescale the space part after normalising.
        scale: fixed scale, or ``None`` with ``use_scale`` for a learnable one.
        learn_scale: make ``scale`` learnable (stored as its log, as upstream).
        manifold_out: optional target manifold.
    """

    def __init__(self, manifold_in, weight=None, batch_size=None, use_scale=False,
                 scale=None, learn_scale=False, manifold_out=None):
        super().__init__()
        self.manifold = manifold_in
        if weight is not None:
            self.w_y = weight
        elif batch_size:
            self.w_y = nn.Parameter(torch.ones((batch_size, 1)))
        else:
            self.w_y = nn.Parameter(torch.tensor(1.0))

        self.scale = None
        self.learned_scale = False
        if use_scale:
            if scale:
                if learn_scale:
                    self.scale = nn.Parameter(torch.tensor(math.log(scale)))
                    self.learned_scale = True
                else:
                    self.scale = scale
            else:
                self.scale = nn.Parameter(torch.tensor(4.0))
                self.learned_scale = True
        self.c = manifold_in.c
        self.manifold_out = manifold_out

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x, y: Lorentz vectors, ``(..., dim)``.
            weight: overrides the module's residual weight.

        Returns:
            The residual, on the manifold.
        """
        w_y = self.w_y if weight is None else weight
        c = self.c
        dim = x.size(-1) - 1

        ave_time = x.narrow(-1, 0, 1) + y.narrow(-1, 0, 1) * w_y
        ave_space = torch.addcmul(x.narrow(-1, 1, dim), y.narrow(-1, 1, dim),
                                  w_y if torch.is_tensor(w_y) else torch.as_tensor(w_y, dtype=x.dtype, device=x.device))

        # <ave, ave>_L = -ave_0^2 + |ave_s|^2; the reduction is reused below.
        sq_space = ave_space.square().sum(dim=-1, keepdim=True)
        denom = (ave_time.square() - sq_space).abs().clamp_min(1e-4).sqrt()

        if self.scale is None:
            out_time = c.sqrt() * ave_time / denom
            out_space = c.sqrt() * ave_space / denom
            out = torch.cat([out_time, out_space], dim=-1)
        else:
            scale = self.scale.exp() if self.learned_scale else self.scale
            coef = (scale * c.sqrt()) / denom
            out_space = coef * ave_space
            # |out_s|^2 = coef^2 * |ave_s|^2 -- no second full-tensor reduction.
            out_time = (coef.square() * sq_space + c).clamp_min(1e-4).sqrt()
            out = torch.cat([out_time, out_space], dim=-1)

        if self.manifold_out is not None:
            out = out * (self.manifold_out.c / c).sqrt()
        return out
