"""End-to-end check that the optimizations do not change what the model learns.

The per-op parity tests in ``test_parity.py`` show the forward and backward
agree at one point. This goes further: it actually trains the reference and the
optimized model side by side from identical initial weights on identical data,
and checks the loss curves and the final weights still agree after many
optimizer steps -- i.e. that small numerical differences do not compound into a
different trajectory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from helm.reference.helm_mice import LorentzDeepSeekV3 as RefModel  # noqa: E402
from tests._config import cast_module, tiny_args  # noqa: E402

STEPS = 60


def make_pair(args, dtype=torch.float64, **fast_kwargs):
    """Reference and optimized models sharing one set of initial weights."""
    torch.manual_seed(0)
    manifolds = (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    ref = cast_module(RefModel(args, *manifolds), dtype)
    fast = HelmMiCE(args, *manifolds, **fast_kwargs)
    fast.load_state_dict(ref.state_dict(), strict=False)
    return ref, cast_module(fast, dtype)


def make_batches(args, steps, seed=1234, batch=2, length=24):
    """A stream with a rule simple enough that the model measurably learns it.

    Each sequence is an arithmetic walk ``t_{i+1} = (t_i + stride) % vocab``, so
    next-token prediction is learnable and the loss actually falls. Random tokens
    would leave the loss pinned near ``ln(vocab)``, and "the curves agree" is a
    much weaker statement when neither curve moves.
    """
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(steps):
        start = torch.randint(0, args.vocab_size, (batch, 1), generator=generator)
        stride = torch.randint(1, 4, (batch, 1), generator=generator)
        offsets = torch.arange(length).unsqueeze(0)
        batches.append((start + stride * offsets) % args.vocab_size)
    return batches


def run_training(model, args, steps=STEPS, lr=5e-3, seed=1234):
    """Train on a fixed synthetic stream; return the loss at each step."""
    torch.manual_seed(seed)
    batches = make_batches(args, steps, seed)

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    for tokens in batches:
        optimizer.zero_grad()
        logits = model(tokens)[0]
        # Standard next-token objective.
        loss = loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                       tokens[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_loss_curves_match_over_many_steps():
    """The two models must follow the same optimization trajectory."""
    args = tiny_args()
    ref, fast = make_pair(args)

    ref_losses = run_training(ref, args)
    fast_losses = run_training(fast, args)

    # Sanity: the comparison is only meaningful if the models are actually
    # learning. "The curves agree" says little when neither curve moves.
    assert ref_losses[-1] < ref_losses[0] - 0.3, (
        f"sanity: the reference should be learning ({ref_losses[0]:.3f} -> "
        f"{ref_losses[-1]:.3f})")

    # The model casts its logits to float32 before the head, so the loss itself
    # only carries float32 resolution even in a float64 model; that sets the
    # floor here, not the optimizations.
    for step, (a, b) in enumerate(zip(ref_losses, fast_losses)):
        assert abs(a - b) < 1e-5, (
            f"loss diverged at step {step}: reference {a!r} vs optimized {b!r}")


def weight_drift(model_a, model_b):
    """Largest absolute parameter difference between two trained models."""
    params_b = dict(model_b.named_parameters())
    return max((param - params_b[name]).abs().max().item()
               for name, param in model_a.named_parameters()
               if name in params_b and not name.endswith("attn.bias"))


def test_trained_weights_stay_within_round_off():
    """The optimized model must not drift further than the reference's own noise.

    A step-0 comparison shows the logits are *bit-identical* and the gradients
    differ only at fp64 machine epsilon (~1e-17), from backward-pass operation
    ordering. Adam amplifies that: it normalises by sqrt(v), so a relative 1e-16
    difference in a small-gradient direction becomes an O(1) difference in the
    update direction, and 60 steps compound it to ~1e-5.

    The tolerance is therefore calibrated rather than guessed. The control is the
    *reference against itself* with one weight moved by a single ULP -- a
    perturbation that is unambiguously "the same model". If the optimized model
    stays within that, the residual is chaotic amplification of round-off, not a
    difference in what the model computes.
    """
    args = tiny_args()

    control_a, _ = make_pair(args)
    control_b, _ = make_pair(args)
    with torch.no_grad():
        weight = control_b.layers[0].attn.wkv_a.linear.weight
        weight[0, 0] = torch.nextafter(
            weight[0, 0], torch.tensor(float("inf"), dtype=weight.dtype))
    run_training(control_a, args)
    run_training(control_b, args)
    round_off = weight_drift(control_a, control_b)

    ref, fast = make_pair(args)
    run_training(ref, args)
    run_training(fast, args)
    observed = weight_drift(ref, fast)

    assert observed <= 2 * round_off, (
        f"optimized model drifted {observed:.2e}, more than twice the "
        f"reference's own {round_off:.2e} response to a one-ULP perturbation")


def test_step_zero_is_bit_identical():
    """Before any optimizer step can amplify anything, nothing differs at all."""
    args = tiny_args()
    ref, fast = make_pair(args, attn_impl="naive", rope_impl="complex",
                          fuse_experts=False, fuse_residual=False)
    tokens = make_batches(args, 1)[0]
    ref.train()
    fast.train()
    ref_logits, ref_idx, _ = ref(tokens)
    fast_logits, fast_idx, _ = fast(tokens)
    assert torch.equal(ref_logits, fast_logits)
    assert all(torch.equal(a, b) for a, b in zip(ref_idx, fast_idx))


def test_frozen_attention_bias_stays_at_zero():
    """The one deliberate parameter difference, and why it is harmless."""
    args = tiny_args()
    ref, fast = make_pair(args)
    run_training(ref, args)
    run_training(fast, args)
    for name, param in fast.named_parameters():
        if name.endswith("attn.bias"):
            # Frozen here; the reference keeps "training" it on softmax
            # round-off, so it wanders off zero without ever changing an output.
            assert param.abs().max() == 0
            assert dict(ref.named_parameters())[name].abs().max() < 1e-4


def test_fused_experts_follow_the_same_trajectory():
    """Fusing the SwiGLU GEMM must not change training either."""
    args = tiny_args()
    _, unfused = make_pair(args, fuse_experts=False)
    _, fused = make_pair(args, fuse_experts=True)

    a = run_training(unfused, args)
    b = run_training(fused, args)
    for step, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < 1e-6, f"loss diverged at step {step}: {x!r} vs {y!r}"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_loss_curves_match_in_float32(dtype):
    """Same check at the precision people actually train at.

    float32 accumulates round-off far faster than float64, so the tolerance is
    looser; the point is that the curves stay together rather than diverging.
    """
    args = tiny_args()
    ref, fast = make_pair(args, dtype=dtype)
    ref_losses = run_training(ref, args)
    fast_losses = run_training(fast, args)
    drift = max(abs(a - b) for a, b in zip(ref_losses, fast_losses))
    spread = max(ref_losses) - min(ref_losses)
    assert drift < 0.01 * spread, (
        f"float32 drift {drift:.2e} is large next to the loss range {spread:.2e}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
