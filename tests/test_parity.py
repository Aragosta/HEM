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
from helm.modules.fused_ce import fused_linear_cross_entropy  # noqa: E402
from helm.modules.lorentz_ops import LorentzResidual  # noqa: E402
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


def rope_table(args, impl, seqlen=None):
    """The rotary table in the layout a given ``rope_impl`` expects."""
    table = precompute_freqs_cis(args) if impl == "complex" else precompute_rope_cache(args)
    return table if seqlen is None else table[:seqlen]


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
@pytest.mark.parametrize("rope_impl", ["complex", "real"])
@pytest.mark.parametrize("use_doc_mask", [False, True])
def test_hmla_matches_reference(attn_impl, rope_impl, use_doc_mask):
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    manifold = Lorentz(1.0)
    ref = RefMLA(manifold, args).double()
    fast = LorentzMLA(manifold, args, attn_impl=attn_impl, rope_impl=rope_impl).double()

    assert set(ref.state_dict()) == set(fast.state_dict()), "state dicts must be interchangeable"
    fast.load_state_dict(ref.state_dict())

    batch, seqlen = 2, 40
    x = on_manifold(batch, seqlen, args.dim - 1)
    causal, doc = masks(batch, seqlen)
    mask = causal.unsqueeze(0) | doc if use_doc_mask else causal

    freqs = precompute_freqs_cis(args)[:seqlen]
    torch.testing.assert_close(fast(x, 0, rope_table(args, rope_impl, seqlen), mask),
                               ref(x, 0, freqs, mask),
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
    torch.testing.assert_close(fast(x, 0, freqs, mask=None, is_causal=True),
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

    def backward(module, table, mask):
        module.zero_grad()
        inp = x.clone().requires_grad_(True)
        module(inp, 0, table, mask).square().sum().backward()
        return inp.grad, {n: p.grad for n, p in module.named_parameters() if p.grad is not None}

    gx_ref, gp_ref = backward(ref, freqs, causal)
    gx_fast, gp_fast = backward(fast, freqs, causal)
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
    table = precompute_freqs_cis(args)

    with torch.no_grad():
        full = fast(x, 0, table[:seqlen], mask=None, is_causal=True)
        from helm.modules.hmla import LorentzKVCache
        kv = LorentzKVCache(batch, seqlen, fast.n_local_heads, fast.qk_head_dim,
                            fast.v_head_dim, dtype=torch.float64, device=x.device)
        steps = [fast(x[:, i:i + 1], i, table[i:i + 1], mask=None, cache=kv)
                 for i in range(seqlen)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------------- residual

@pytest.mark.parametrize("config", [
    pytest.param(dict(use_scale=True, scale=19.75, learn_scale=False), id="block"),
    pytest.param(dict(use_scale=True, scale=2.0, learn_scale=True), id="moe-add"),
    pytest.param(dict(weight=1.0, use_scale=True, scale=2.0, learn_scale=False),
                 id="moe-weighted"),
    pytest.param(dict(), id="no-scale"),
])
@pytest.mark.parametrize("per_row_weight", [False, True])
def test_fused_residual_matches_lresnet(config, per_row_weight):
    """The fused residual must equal HyperCore's LResNet in every configuration."""
    from helm.hypercore.nn.conv.conv_util_layers import LResNet

    torch.manual_seed(0)
    manifold = Lorentz(1.0)
    ref = LResNet(manifold, **config).double()
    fast = LorentzResidual(manifold, **config).double()
    assert set(ref.state_dict()) == set(fast.state_dict())
    fast.load_state_dict(ref.state_dict())

    x = on_manifold(2, 24, 32)
    y = on_manifold(2, 24, 32)
    weight = torch.rand(2, 24, 1, dtype=torch.float64) if per_row_weight else None
    torch.testing.assert_close(fast(x, y, weight), ref(x, y, weight),
                               rtol=1e-12, atol=1e-13)


def test_fused_residual_gradients_match():
    from helm.hypercore.nn.conv.conv_util_layers import LResNet

    torch.manual_seed(0)
    manifold = Lorentz(1.0)
    ref = LResNet(manifold, use_scale=True, scale=19.75, learn_scale=False).double()
    fast = LorentzResidual(manifold, use_scale=True, scale=19.75, learn_scale=False).double()
    fast.load_state_dict(ref.state_dict())

    x0 = on_manifold(2, 24, 32)
    y0 = on_manifold(2, 24, 32)
    grads = []
    for module in (ref, fast):
        x = x0.clone().requires_grad_(True)
        y = y0.clone().requires_grad_(True)
        module.zero_grad()
        module(x, y).square().sum().backward()
        grads.append((x.grad, y.grad, module.w_y.grad))
    for a, b in zip(grads[0], grads[1]):
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dtype_cast_preserves_rotary_table(dtype):
    """``.to(dtype)`` must not rewrite the rotary table.

    The upstream model stores it as a complex tensor, so casting the module to
    any real dtype discards ``sin`` and leaves rotary embeddings as a cosine
    rescale -- a warning on stderr and a quietly broken model.
    """
    args = tiny_args()
    manifolds = (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))

    ref = RefModel(args, *manifolds)
    ref.to(dtype)
    assert ref.freqs_cis.is_complex() is False, "upstream is expected to lose its table"

    for rope_impl in ("real", "complex"):
        model = HelmMiCE(args, *manifolds, rope_impl=rope_impl)
        before = model.freqs_cis.clone()
        model.to(dtype)
        assert model.freqs_cis.dtype == before.dtype
        assert torch.equal(model.freqs_cis, before)


@pytest.mark.parametrize("mode", ["latent", "naive"])
def test_model_eval_and_generation(mode):
    """Full model in eval mode, and cached decoding equal to a full forward."""
    torch.manual_seed(0)
    args = tiny_args()
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)).double().eval()
    tokens = torch.randint(0, args.vocab_size, (2, 10))
    with torch.no_grad():
        full = model(tokens)
        caches = model.new_kv_caches(2, args.max_seq_len, dtype=torch.float64, mode=mode)
        steps = [model(tokens[:, i:i + 1], start_pos=i, caches=caches)
                 for i in range(tokens.size(1))]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["latent", "naive"])
def test_prefill_then_decode_matches_full_forward(mode):
    """Prefill a chunk into the cache, then decode -- the mask has to stay causal.

    Feeding one token at a time never exercises a multi-token query block against
    a cache. With ``is_causal`` switched off whenever a cache was present (as it
    briefly was here), the prefill block attended *bidirectionally* and every
    later layer cached keys derived from the wrong hidden states -- invisible to
    a token-at-a-time test, and invisible to greedy decoding, which only reads
    the last position.
    """
    torch.manual_seed(0)
    args = tiny_args()
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)).double().eval()
    tokens = torch.randint(0, args.vocab_size, (2, 12))
    split = 7
    with torch.no_grad():
        full = model(tokens)
        caches = model.new_kv_caches(2, args.max_seq_len, dtype=torch.float64, mode=mode)
        prefill = model(tokens[:, :split], start_pos=0, caches=caches)
        decoded = [model(tokens[:, i:i + 1], start_pos=i, caches=caches)
                   for i in range(split, tokens.size(1))]
    got = torch.cat([prefill] + decoded, dim=1)
    torch.testing.assert_close(got, full, rtol=1e-5, atol=1e-6)


def test_latent_cache_is_smaller_than_naive():
    """MLA's whole point: cache the latent, not the reconstructed heads."""
    from helm.eval.presets import preset_args

    for name, expected_ratio in (("helm_mice_120M", 6.0), ("helm_mice_1B", 14.0)):
        args = preset_args(name, n_layers=1, vocab_size=256, max_seq_len=128,
                           original_seq_len=128)
        model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)).eval()
        latent = model.new_kv_caches(1, 128, mode="latent")[0].numel()
        naive = model.new_kv_caches(1, 128, mode="naive")[0].numel()
        assert naive / latent > expected_ratio, (name, naive, latent, naive / latent)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_exact_configuration_is_bit_identical():
    """Every optimization can be turned off individually, back to bit-exactness.

    ``attn_impl="naive"``, ``rope_impl="complex"``, ``fuse_experts=False`` and
    ``fuse_residual=False`` select the literal published formulation of each
    piece. The scheduling changes that remain -- the sorted MoE dispatch, the
    frozen bias, the mask handling, the non-persistent buffers -- must then
    produce output that is bit-for-bit identical to the reference.
    """
    torch.manual_seed(0)
    args = tiny_args()
    manifolds = (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    ref = RefModel(args, *manifolds).double()
    fast = HelmMiCE(args, *manifolds, attn_impl="naive", rope_impl="complex",
                    fuse_experts=False, fuse_residual=False).double()
    fast.load_state_dict(ref.state_dict(), strict=False)

    tokens = torch.randint(0, args.vocab_size, (2, 24))
    ref.train()
    fast.train()
    ref_logits, ref_idx, ref_scores = ref(tokens)
    fast_logits, fast_idx, fast_scores = fast(tokens)
    assert torch.equal(ref_logits, fast_logits)
    assert all(torch.equal(a, b) for a, b in zip(ref_idx, fast_idx))
    assert all(torch.equal(a, b) for a, b in zip(ref_scores, fast_scores))


# ------------------------------------------------------------------ fused head

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_bias", [False, True])
def test_fused_cross_entropy_matches_reference(dtype, use_bias):
    """Fused head == materialising logits, casting to fp32, and taking CE."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    batch, seqlen, dim, vocab = 2, 40, 16, 97
    hidden = torch.randn(batch, seqlen, dim, dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab, dim, dtype=dtype, requires_grad=True)
    bias = torch.randn(vocab, dtype=dtype, requires_grad=True) if use_bias else None

    target = torch.randint(0, vocab, (batch, seqlen))
    target[:, -1] = -100                    # shifted-label padding
    target[target % 5 == 0] = -100          # sequence-packing gaps

    reference = F.cross_entropy(
        F.linear(hidden, weight, bias).float().reshape(-1, vocab),
        target.reshape(-1), ignore_index=-100)
    fused = fused_linear_cross_entropy(hidden, weight, target, bias, chunk_size=7)

    torch.testing.assert_close(fused, reference.float(), rtol=1e-6, atol=1e-6)

    inputs = [hidden, weight] + ([bias] if use_bias else [])
    for got, want in zip(torch.autograd.grad(fused, inputs),
                         torch.autograd.grad(reference, inputs)):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_fused_cross_entropy_chunk_size_is_irrelevant():
    """Chunking is a scheduling choice; it must not change the answer."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 33, 16, dtype=torch.float64)
    weight = torch.randn(97, 16, dtype=torch.float64)
    target = torch.randint(0, 97, (2, 33))
    losses = [fused_linear_cross_entropy(hidden, weight, target, chunk_size=c).item()
              for c in (1, 7, 32, 4096)]
    assert max(losses) - min(losses) == 0.0, losses


def test_fused_cross_entropy_all_ignored():
    """A fully-masked batch yields zero loss, not a NaN from dividing by zero."""
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    weight = torch.randn(5, 8, requires_grad=True)
    loss = fused_linear_cross_entropy(hidden, weight, torch.full((2, 4), -100))
    assert loss.item() == 0.0 and torch.isfinite(loss)


def test_model_fused_head_matches_logits_path():
    """The model's ``labels=`` path must equal logits + CrossEntropyLoss."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    args = tiny_args()
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)).double()
    tokens = torch.randint(0, args.vocab_size, (2, 24))
    labels = tokens.roll(-1, 1).clone()
    labels[:, -1] = -100
    labels[labels % 5 == 0] = -100
    model.train()

    logits, _, _ = model(tokens)
    reference = F.cross_entropy(logits.reshape(-1, args.vocab_size),
                                labels.reshape(-1), ignore_index=-100)
    model.zero_grad()
    reference.backward()
    ref_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                 if p.grad is not None}

    fused, _, _ = model(tokens, labels=labels)
    model.zero_grad()
    fused.backward()
    fused_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                   if p.grad is not None}

    torch.testing.assert_close(fused, reference.float(), rtol=1e-6, atol=1e-6)
    assert set(ref_grads) == set(fused_grads)
    for name in ref_grads:
        torch.testing.assert_close(fused_grads[name], ref_grads[name],
                                   rtol=1e-5, atol=1e-6,
                                   msg=lambda m, n=name: f"{n}: {m}")
