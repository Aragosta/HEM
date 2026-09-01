"""Small HELM-MiCE configs used by the parity tests and the benchmarks."""

from types import SimpleNamespace


def tiny_args(**overrides):
    """A ~1M-parameter HELM-MiCE config that runs in a second on CPU.

    Shapes follow ``example/train_mice_120M.sh`` (which is the smallest released
    configuration) scaled down; every structural relationship that matters --
    ``dim`` odd so ``dim - 1`` is even, rope dim odd, kv_lora_rank odd -- is
    preserved.
    """
    args = SimpleNamespace(
        # training / shape
        max_batch_size=2,
        max_seq_len=32,
        vocab_size=97,
        project_emb=0,
        train=True,
        # model
        dim=33,
        inter_dim=64,
        moe_inter_dim=48,
        mice_inter_dim=48,
        n_layers=3,
        n_dense_layers=1,
        n_heads=3,
        # MiCE
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        score_func="softmax",
        route_scale=1.0,
        bias_update_speed=0.005,
        seq_bal_alpha=1e-4,
        train_curv=True,
        # HMLA
        q_lora_rank=0,
        kv_lora_rank=17,
        qk_nope_head_dim=9,
        qk_rope_head_dim=9,
        v_head_dim=9,
        # yarn
        original_seq_len=32,
        rope_theta=10000,
        rope_factor=40,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def bench_args(**overrides):
    """The released 120M HELM-MiCE shape (``example/train_mice_120M.sh``)."""
    return tiny_args(
        max_batch_size=4,
        max_seq_len=2048,
        original_seq_len=2048,
        vocab_size=128256,
        dim=390,
        inter_dim=1560,
        moe_inter_dim=780,
        mice_inter_dim=780,
        n_layers=6,
        n_dense_layers=1,
        n_heads=6,
        n_routed_experts=4,
        kv_lora_rank=65,
        qk_nope_head_dim=33,
        qk_rope_head_dim=17,
        v_head_dim=33,
        **overrides,
    )
