"""Numerical parity between the optimized HELM-MiCE and the published reference.

Every optimization in :mod:`helm.modules` is an algebraic rewrite, not an
approximation, so the two implementations must agree to floating-point noise.
These tests pin that down, in float64, against the untouched upstream code in
:mod:`helm.reference`.

Run with ``python -m pytest tests/ -v`` (or ``python tests/test_parity.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from helm.modules.hmla import LorentzMLA  # noqa: E402
from helm.modules.mice import LorentzMoE  # noqa: E402
from helm.modules.rope import (apply_rotary_emb, apply_rotary_emb_real,  # noqa: E402
                               precompute_freqs_cis, precompute_rope_cache)
from helm.reference.helm_mice import LorentzDeepSeekV3 as RefModel  # noqa: E402
from helm.reference.hmla import LorentzMLA as RefMLA  # noqa: E402
from helm.reference.mice import LorentzMoE as RefMoE  # noqa: E402
from tests._config import tiny_args  # noqa: E402

# The reference casts its attention softmax to float32 unconditionally
# (`scores.softmax(dim=-1, dtype=torch.float32)`), so even a float64 model is
# only accurate to ~1e-7 there. The optimized path keeps full precision, which
# means "agrees with the reference" is bounded by the reference, not by us.
ATOL_REF_SOFTMAX = 1e-5


def on_manifold(*shape, dtype=torch.float64, scale=0.3):
    """A batch of random points on the unit-curvature hyperboloid."""
    space = torch.randn(*shape, dtype=dtype) * scale
    time = (space.square().sum(-1, keepdim=True) + 1).sqrt()
    return torch.cat([time, space], dim=-1)


def masks(batch, seqlen):
    causal = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool), 1)
    seq = torch.zeros(batch, seqlen, dtype=torch.long)
    seq[:, seqlen // 2:] = 1
    doc = ~(seq.unsqueeze(1) == seq.unsqueeze(2))
    return causal, doc


# --------------------------------------------------------------------- rotary

def test_rotary_real_matches_complex():
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    freqs = precompute_freqs_cis(args)[:20]
    cache = precompute_rope_cache(args)[:20]
    x = torch.randn(2, 20, 3, args.qk_rope_head_dim - 1)
    torch.testing.assert_close(apply_rotary_emb(x, freqs),
                               apply_rotary_emb_real(x, cache),
                               rtol=0, atol=1e-6)


# ------------------------------------------------------------------ attention

@pytest.mark.parametrize("attn_impl", ["naive", "flash"])
@pytest.mark.parametrize("use_doc_mask", [False, True])
def test_hmla_matches_reference(attn_impl, use_doc_mask):
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    manifold = Lorentz(1.0)
    ref = RefMLA(manifold, args).double()
    fast = LorentzMLA(manifold, args, attn_impl=attn_impl).double()

    assert set(ref.state_dict()) == set(fast.state_dict()), "state dicts must be interchangeable"
    fast.load_state_dict(ref.state_dict())

    batch, seqlen = 2, 40
    x = on_manifold(batch, seqlen, args.dim - 1)
    causal, doc = masks(batch, seqlen)
    mask = causal.unsqueeze(0) | doc if use_doc_mask else causal

    freqs = precompute_freqs_cis(args)[:seqlen]
    cache = precompute_rope_cache(args)[:seqlen]
    torch.testing.assert_close(fast(x, 0, cache, mask), ref(x, 0, freqs, mask),
                               rtol=0, atol=ATOL_REF_SOFTMAX)


def test_hmla_naive_complex_is_bit_exact():
    """With the same softmax precision and the same rotary path, nothing moves.

    This is the strongest statement available: dropping the constant ``2c`` and
    ``bias`` terms and folding the temperature into the query is an exact
    rewrite, not an approximation.
    """
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    manifold = Lorentz(1.0)
    ref = RefMLA(manifold, args).double()
    fast = LorentzMLA(manifold, args, attn_impl="naive", rope_impl="complex").double()
    fast.load_state_dict(ref.state_dict())

    batch, seqlen = 2, 40
    x = on_manifold(batch, seqlen, args.dim - 1)
    causal, _ = masks(batch, seqlen)
    freqs = precompute_freqs_cis(args)[:seqlen]
    assert torch.equal(fast(x, 0, freqs, causal), ref(x, 0, freqs, causal))


def test_hmla_is_causal_flag_matches_explicit_mask():
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    manifold = Lorentz(1.0)
    ref = RefMLA(manifold, args).double()
    fast = LorentzMLA(manifold, args).double()
    fast.load_state_dict(ref.state_dict())

    batch, seqlen = 2, 40
    x = on_manifold(batch, seqlen, args.dim - 1)
    causal, _ = masks(batch, seqlen)
    freqs = precompute_freqs_cis(args)[:seqlen]
    cache = precompute_rope_cache(args)[:seqlen]
    torch.testing.assert_close(fast(x, 0, cache, mask=None, is_causal=True),
                               ref(x, 0, freqs, causal),
                               rtol=0, atol=ATOL_REF_SOFTMAX)


def test_hmla_gradients_match_reference():
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    manifold = Lorentz(1.0)
    ref = RefMLA(manifold, args).double()
    fast = LorentzMLA(manifold, args).double()
    fast.load_state_dict(ref.state_dict())

    batch, seqlen = 2, 40
    x = on_manifold(batch, seqlen, args.dim - 1)
    causal, _ = masks(batch, seqlen)
    freqs = precompute_freqs_cis(args)[:seqlen]
    cache = precompute_rope_cache(args)[:seqlen]

    def backward(module, table, mask):
        module.zero_grad()
        inp = x.clone().requires_grad_(True)
        module(inp, 0, table, mask).square().sum().backward()
        return inp.grad, {n: p.grad for n, p in module.named_parameters() if p.grad is not None}

    gx_ref, gp_ref = backward(ref, freqs, causal)
    gx_fast, gp_fast = backward(fast, cache, causal)
    torch.testing.assert_close(gx_fast, gx_ref, rtol=1e-4, atol=1e-4)
    for name, grad in gp_fast.items():
        torch.testing.assert_close(grad, gp_ref[name], rtol=1e-4, atol=1e-4,
                                   msg=lambda m, n=name: f"{n}: {m}")


def test_attention_bias_has_no_effect():
    """The score bias is a per-row constant, so the softmax cancels it exactly.

    The optimized path drops the term and freezes the parameter. This shows
    nothing is lost: the reference's own output is invariant to the bias, and the
    gradient it reports is pure float32 softmax round-off (the reference softmaxes
    in fp32 regardless of model dtype), orders of magnitude below a real signal.
    """
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    ref = RefMLA(Lorentz(1.0), args).double()
    x = on_manifold(2, 40, args.dim - 1)
    causal, _ = masks(2, 40)
    freqs = precompute_freqs_cis(args)[:40]

    with torch.no_grad():
        baseline = ref(x, 0, freqs, causal)
        ref.bias.fill_(7.5)
        shifted = ref(x, 0, freqs, causal)
        ref.bias.zero_()
    # The residual is the reference's own fp32 softmax cast, not a real
    # dependence on the bias: an fp64 softmax cancels it to machine precision.
    torch.testing.assert_close(shifted, baseline, rtol=1e-4, atol=1e-5)

    ref(x, 0, freqs, causal).square().sum().backward()
    typical = max(p.grad.abs().max().item() for n, p in ref.named_parameters()
                  if p.grad is not None and n != "bias")
    assert ref.bias.grad.abs().max() < 1e-8 * typical, \
        "bias gradient should be round-off, not signal"


def test_kv_cache_matches_full_forward():
    """Decoding token by token with the cache equals one full-prefix forward."""
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    manifold = Lorentz(1.0)
    fast = LorentzMLA(manifold, args).double().eval()

    batch, seqlen = 2, 12
    x = on_manifold(batch, seqlen, args.dim - 1)
    cache_table = precompute_rope_cache(args).double()

    with torch.no_grad():
        full = fast(x, 0, cache_table[:seqlen], mask=None, is_causal=True)
        kv = [torch.zeros(0)]  # placeholder, replaced below
        from helm.modules.hmla import LorentzKVCache
        kv = LorentzKVCache(batch, seqlen, fast.n_local_heads, fast.qk_head_dim,
                            fast.v_head_dim, dtype=torch.float64, device=x.device)
        steps = [fast(x[:, i:i + 1], i, cache_table[i:i + 1], mask=None, cache=kv)
                 for i in range(seqlen)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------------------ MoE

@pytest.mark.parametrize("fuse_experts", [False, True])
def test_moe_matches_reference(fuse_experts):
    torch.manual_seed(0)
    args = tiny_args()
    manifold = Lorentz(1.0)
    ref = RefMoE(manifold, args).double()
    fast = LorentzMoE(manifold, args, fuse_experts=fuse_experts).double()
    missing, unexpected = fast.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    x = on_manifold(40, args.dim - 1)
    ref.train()
    fast.train()
    out_ref, idx_ref, scores_ref = ref(x)
    out_fast, idx_fast, scores_fast = fast(x)
    assert torch.equal(idx_ref, idx_fast), "routing decisions must be identical"
    torch.testing.assert_close(scores_fast, scores_ref, rtol=0, atol=1e-12)
    torch.testing.assert_close(out_fast, out_ref, rtol=1e-10, atol=1e-12)


def test_moe_eval_mode_works():
    """Upstream cannot run a MiCE layer outside training mode; this one can."""
    args = tiny_args()
    manifold = Lorentz(1.0)
    x = on_manifold(8, args.dim - 1)

    ref = RefMoE(manifold, args).double().eval()
    with pytest.raises(ValueError):
        ref(x)

    fast = LorentzMoE(manifold, args).double().eval()
    assert fast(x).shape == x.shape


# ---------------------------------------------------------------- whole model

@pytest.mark.parametrize("use_doc_mask", [False, True])
def test_model_matches_reference(use_doc_mask):
    torch.manual_seed(0)
    args = tiny_args()
    manifolds = (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    ref = RefModel(args, *manifolds).double()
    fast = HelmMiCE(args, *manifolds).double()
    missing, unexpected = fast.load_state_dict(ref.state_dict(), strict=False)
    assert not missing, missing
    assert unexpected == ["attn_mask"], unexpected  # dropped on purpose

    batch, seqlen = 2, 24
    tokens = torch.randint(0, args.vocab_size, (batch, seqlen))
    doc = None
    if use_doc_mask:
        seq = torch.zeros(batch, seqlen, dtype=torch.long)
        seq[:, seqlen // 2:] = 1
        doc = ~(seq.unsqueeze(1) == seq.unsqueeze(2))

    ref.train()
    fast.train()
    logits_ref, idx_ref, _ = ref(tokens, attn_mask=doc)
    logits_fast, idx_fast, _ = fast(tokens, attn_mask=doc)
    torch.testing.assert_close(logits_fast, logits_ref, rtol=1e-4, atol=1e-3)
    for a, b in zip(idx_ref, idx_fast):
        assert torch.equal(a, b)


def test_model_grad_checkpointing_matches():
    torch.manual_seed(0)
    args = tiny_args()
    manifolds = (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    plain = HelmMiCE(args, *manifolds).double()
    ckpt = HelmMiCE(args, *manifolds, grad_checkpoint=True).double()
    ckpt.load_state_dict(plain.state_dict())

    tokens = torch.randint(0, args.vocab_size, (2, 24))
    grads = []
    for model in (plain, ckpt):
        model.train()
        model.zero_grad()
        model(tokens)[0].square().sum().backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters()
                      if p.grad is not None})
    assert set(grads[0]) == set(grads[1])
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-9, atol=1e-12,
                                   msg=lambda m, n=name: f"{n}: {m}")


def test_model_eval_and_generation():
    """Full model in eval mode, and cached decoding equal to a full forward."""
    torch.manual_seed(0)
    args = tiny_args()
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)).double().eval()
    tokens = torch.randint(0, args.vocab_size, (2, 10))
    with torch.no_grad():
        full = model(tokens)
        caches = model.new_kv_caches(2, args.max_seq_len, dtype=torch.float64)
        steps = [model(tokens[:, i:i + 1], start_pos=i, caches=caches)
                 for i in range(tokens.size(1))]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
