"""Released HELM model shapes.

Transcribed from the model config classes in the authors' fork of
``lm-evaluation-harness`` (``lm_eval/models/helm.py``), which is where the
evaluation numbers in the paper come from. Note these are the *evaluation*
shapes; a couple of fields disagree with the corresponding ``example/train_*.sh``
script -- see ``KNOWN_TRAIN_EVAL_MISMATCHES``.
"""

from __future__ import annotations

from types import SimpleNamespace

_COMMON = dict(
    max_batch_size=8,
    max_seq_len=2048,
    vocab_size=128256,
    n_dense_layers=1,
    n_expert_groups=1,
    n_limited_groups=1,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="softmax",
    route_scale=1.0,
    bias_update_speed=0.005,
    seq_bal_alpha=1e-4,
    train_curv=False,
    q_lora_rank=0,
    original_seq_len=2048,
    rope_theta=10000.0,
    rope_factor=40,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
)

PRESETS = {
    "helm_mice_120M": dict(
        _COMMON,
        model_name="HELM_MiCE",
        dim=390,
        inter_dim=390 * 4,
        moe_inter_dim=780,
        n_layers=6,
        n_heads=6,
        n_routed_experts=4,
        kv_lora_rank=65,
        qk_nope_head_dim=33,
        qk_rope_head_dim=17,
        v_head_dim=33,
        project_emb=False,
    ),
    "helm_mice_1B": dict(
        _COMMON,
        model_name="HELM_MiCE",
        dim=64 * 14,
        inter_dim=12 * 64 * 4,
        moe_inter_dim=14 * 64 * 2,
        n_layers=15,
        n_heads=14,
        n_routed_experts=8,
        kv_lora_rank=257,
        qk_nope_head_dim=129,
        qk_rope_head_dim=65,
        v_head_dim=129,
        project_emb=True,
    ),
    "helm_d_115M": dict(
        _COMMON,
        model_name="HELM_D",
        arch="L6_W390_A6",
        dim=390,
        inter_dim=390 * 4,
        moe_inter_dim=780,
        n_layers=6,
        n_heads=6,
        n_routed_experts=4,
        kv_lora_rank=65,
        qk_nope_head_dim=33,
        qk_rope_head_dim=17,
        v_head_dim=33,
        project_emb=False,
    ),
}

#: Fields where the released evaluation config disagrees with the training
#: script of the same name. Recorded rather than silently reconciled -- a
#: checkpoint has to be loaded with the shape it was trained at, and only the
#: authors can say which is authoritative.
KNOWN_TRAIN_EVAL_MISMATCHES = {
    "helm_mice_1B": {
        # example/train_mice_1B.sh passes 65/65, the eval config uses 129/65.
        "qk_nope_head_dim": {"train": 65, "eval": 129},
        "v_head_dim": {"train": 65, "eval": 129},
        # train_mice_1B.sh runs 16 layers, the eval config 15.
        "n_layers": {"train": 16, "eval": 15},
    },
}


def preset_args(name: str, **overrides) -> SimpleNamespace:
    """Build a model-config namespace for a released HELM shape.

    Args:
        name: one of :data:`PRESETS`.
        **overrides: fields to change (e.g. ``max_seq_len``).

    Returns:
        A namespace accepted by :class:`helm.modules.helm_mice.HelmMiCE`.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    config = dict(PRESETS[name])
    config.update(overrides)
    config.setdefault("mice_inter_dim", config["moe_inter_dim"])
    return SimpleNamespace(**config)
