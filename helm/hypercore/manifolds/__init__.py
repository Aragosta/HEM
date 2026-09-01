"""Manifolds.

Resolved lazily (see :mod:`helm._lazy`): the Lorentz manifold used by the
language model should not require ``torchvision``, which the pseudo-hyperboloid
module imports.
"""

from helm._lazy import install_lazy

_EXPORTS = {
    "Lorentz": ".lorentzian",
    "PoincareBall": ".poincare",
    "PseudoHyperboloid": ".pseudohyperboloid_sr",
}

install_lazy(__name__, globals(), _EXPORTS)
