"""Tests for the training-loop pieces that the released script got wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helm.modules.mice import Gate  # noqa: E402
from tests._config import tiny_args  # noqa: E402


def load_balance_loss():
    """Import ``sequence_balance_loss`` from train.py without its heavy imports.

    ``train.py`` pulls in accelerate/llmfoundry/transformers at module scope, so
    the function is lifted out by source rather than imported.
    """
    source = (ROOT / "train.py").read_text()
    start = source.index("def sequence_balance_loss")
    end = source.index("\ndef build_model")
    namespace = {"torch": torch}
    exec(compile(source[start:end], "train.py", "exec"), namespace)
    return namespace["sequence_balance_loss"]


def test_balance_loss_is_minimised_by_uniform_routing():
    """The auxiliary loss must actually penalise imbalance."""
    balance = load_balance_loss()
    n_tokens, n_experts, topk = 64, 4, 2

    uniform_idx = torch.arange(n_tokens * topk).reshape(n_tokens, topk) % n_experts
    uniform_scores = torch.full((n_tokens, n_experts), 1.0 / n_experts)

    collapsed_idx = torch.zeros(n_tokens, topk, dtype=torch.long)
    collapsed_idx[:, 1] = 1
    collapsed_scores = torch.zeros(n_tokens, n_experts)
    collapsed_scores[:, :2] = 0.5

    assert balance(collapsed_scores, collapsed_idx, 1.0) > balance(uniform_scores, uniform_idx, 1.0)


def test_balance_loss_shape_independent_of_topk():
    """Upstream's version only broadcasts when topk happens to equal n_experts."""
    balance = load_balance_loss()
    for topk in (1, 2, 3):
        idx = torch.randint(0, 8, (32, topk))
        scores = torch.rand(32, 8).softmax(-1)
        out = balance(scores, idx, 1e-4)
        assert out.shape == () and torch.isfinite(out)


def test_balance_loss_stays_on_device():
    balance = load_balance_loss()
    scores = torch.rand(16, 4).softmax(-1)
    idx = torch.randint(0, 4, (16, 2))
    assert balance(scores, idx, 1e-4).device == scores.device


def test_upstream_balance_loss_is_broken():
    """Pin the bug this replaces, so the rewrite is justified in the record."""
    def upstream(scores, indices, alpha):
        N, E = scores.size()
        k = 2
        indices = indices.type_as(scores)
        freq = torch.bincount(indices.flatten().long(), minlength=E).float()
        freq = indices * (E / (k * N))          # overwrites the histogram
        probs = scores / scores.sum(dim=-1, keepdim=True)
        P = probs.mean(dim=0)
        return alpha * (freq * P).sum()

    scores = torch.rand(32, 8).softmax(-1)
    with pytest.raises(RuntimeError):           # (32, 2) cannot broadcast to (8,)
        upstream(scores, torch.randint(0, 8, (32, 2)), 1e-4)


def test_gate_bias_update_moves_towards_uniform():
    """`update_bias` must penalise over-used experts and reward starved ones."""
    args = tiny_args(n_routed_experts=4, bias_update_speed=0.1)
    gate = Gate(args)
    # Expert 0 takes every token; experts 1-3 take none.
    gate.update_bias(torch.zeros(16, 2, dtype=torch.long))
    assert gate.bias[0] < 0, "over-used expert should be discouraged"
    assert (gate.bias[1:] > 0).all(), "starved experts should be encouraged"


def test_gate_bias_update_accepts_a_histogram():
    """The training loop accumulates counts on device and passes them straight in."""
    args = tiny_args(n_routed_experts=4, bias_update_speed=0.1)
    gate = Gate(args)
    from_indices = Gate(args)
    counts = torch.tensor([32.0, 0.0, 0.0, 0.0])
    gate.update_bias(counts)
    from_indices.update_bias(torch.zeros(16, 2, dtype=torch.long))
    torch.testing.assert_close(gate.bias, from_indices.bias)


def test_gate_bias_is_not_trained_by_the_optimizer():
    """It is maintained by `update_bias`; weight decay must not drag it around."""
    gate = Gate(tiny_args())
    assert not gate.bias.requires_grad


def test_boolean_cli_flags_round_trip():
    """`--train_curv True` crashed upstream (`int("True")`); every example uses it."""
    from config.args import parser

    args = parser.parse_args(["--train_curv", "True", "--project_emb", "False"])
    assert args.train_curv is True
    assert args.project_emb is False
    args = parser.parse_args(["--train_curv", "false"])
    assert args.train_curv is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
