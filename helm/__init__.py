"""HELM: Hyperbolic Large Language Models.

Port of https://github.com/Graph-and-Geometric-Learning/helm (Apache-2.0) with
an optimized HELM-MiCE implementation. See ``docs/OPTIMIZATIONS.md``.

Subpackages are resolved lazily: importing the language model should not drag in
the graph-learning stack (``torch_geometric``/``torch_scatter``) that the rest of
HyperCore pulls in. ``import helm; helm.hypercore`` still works.
"""

from helm._lazy import install_lazy

install_lazy(__name__, globals(), {}, submodules=("hypercore", "modules", "utils", "reference"))
