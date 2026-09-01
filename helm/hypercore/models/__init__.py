"""Reference HyperCore models.

Resolved lazily (see :mod:`helm._lazy`) so that importing ``LorentzFeedForward``
for the language model does not also import the graph models (``sklearn``) or the
vision models (``torchvision``).
"""

from helm._lazy import install_lazy

_EXPORTS = {
    "BaseModel": ".graph_models",
    "LCLIP": ".LCLIP",
    "LPModel": ".graph_models",
    "LTransformerEncoder": ".Transformer_encoder",
    "LViT": ".LViT",
    "LorentzFeedForward": ".lorentz_feedforward",
    "Lorentz_ResNet": ".lorentz_resnet",
    "Lorentz_resnet101": ".lorentz_resnet",
    "Lorentz_resnet152": ".lorentz_resnet",
    "Lorentz_resnet18": ".lorentz_resnet",
    "Lorentz_resnet34": ".lorentz_resnet",
    "Lorentz_resnet50": ".lorentz_resnet",
    "MDModel": ".graph_models",
    "NCModel": ".graph_models",
    "Tokenizer": ".tokenizer",
}

install_lazy(__name__, globals(), _EXPORTS)
