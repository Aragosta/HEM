"""HyperCore: hyperbolic building blocks (vendored from the HELM release).

Subpackages are imported lazily so that pulling in a single Lorentz layer does
not also import the graph (``torch_geometric``), vision (``torchvision``) and
metrics (``sklearn``) stacks. ``hypercore.nn``, ``hypercore.manifolds``, ... all
still resolve on first attribute access.
"""

from helm._lazy import install_lazy

install_lazy(
    __name__,
    globals(),
    {},
    submodules=("utils", "nn", "manifolds", "models", "optimizers", "datasets"),
)
