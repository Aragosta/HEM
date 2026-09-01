"""Lazy submodule re-exports (PEP 562).

Upstream, every ``__init__.py`` in :mod:`helm.hypercore` eagerly imported every
submodule underneath it. Importing the language model therefore also imported
the graph-learning stack (``torch_geometric``, ``torch_scatter``), the vision
stack (``torchvision``) and ``sklearn`` -- several seconds of start-up per
process (multiplied by every dataloader worker and every rank) and a much
heavier dependency set than an LM run actually needs.

``install_lazy`` keeps the exact same attribute surface (``hnn.LorentzLinear``
still resolves) but defers the underlying import to first use.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Iterable


def install_lazy(
    package: str,
    namespace: dict,
    exports: Dict[str, str],
    submodules: Iterable[str] = (),
):
    """Wire up ``__getattr__``/``__dir__`` for a package.

    Args:
        package: ``__name__`` of the calling package.
        namespace: the calling module's ``globals()``.
        exports: attribute name -> relative submodule that defines it.
        submodules: submodules exposed as attributes under their own name.
    """
    submodules = tuple(submodules)
    all_names = sorted(set(exports) | set(submodules))

    def __getattr__(name: str):
        if name in submodules:
            module = importlib.import_module(f".{name}", package)
            namespace[name] = module
            return module
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module: ModuleType = importlib.import_module(target, package)
        value = getattr(module, name)
        namespace[name] = value
        return value

    def __dir__():
        return all_names

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
    namespace["__all__"] = all_names
