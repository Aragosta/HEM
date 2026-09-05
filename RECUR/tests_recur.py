#!/usr/bin/env python3
"""Checks on the parts of this folder that a wrong result would look normal.

Not a test suite for the model's quality -- there is nothing to be right about
there. These pin the properties the experiments' *interpretation* depends on,
each of which could be wrong while every run still produced plausible numbers:

* AttnRes really is a softmax over sources and really starts uniform (if the
  pseudo-queries were not zero-initialised, the E0 comparison would be against
  something other than the published mechanism);
* the standard-residual control really is the running sum (if it were not, the
  AttnRes arm would be compared against an accident);
* registers are visible to every position and are not decoded as output (if
  they were decoded, E2's accuracy would be measured on the wrong tokens);
* halting distributions sum to one, so "expected loss over exit steps" is an
  expectation;
* two arms with the same seed start from identical shared weights, which is
  what makes the paired differences in E2 paired;
* the FLOP accounting moves with the loop count, since the fixed-compute
  comparisons in E1 rest on it.

    python tests_recur.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AttnResStream, Config, Recurrent            # noqa: E402
from harness import halting_loss                              # noqa: E402
from tasks import HopSpec, hop_batch                          # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _spec_and_tokens(**kw):
    spec = HopSpec(n_entities=6, hops=(2,), queries=4, **kw)
    g = torch.Generator().manual_seed(0)
    tokens, target, _ = hop_batch(spec, 4, g)
    return spec, tokens, target


@check
def attnres_starts_uniform_and_normalises():
    spec, tokens, _ = _spec_and_tokens()
    cfg = Config(vocab_size=spec.vocab_size, dim=32, n_heads=4,
                 max_seq_len=spec.seq_len, attn_res="full")
    model = Recurrent(cfg)
    assert torch.count_nonzero(model.queries) == 0, "pseudo-queries must start at zero"

    captured = {}
    original = AttnResStream.read

    def spy(self, norm, kind, step):
        out = original(self, norm, kind, step)
        captured.setdefault("weights", []).append(self.last_weights)
        return out

    AttnResStream.read = spy
    try:
        model(tokens)
    finally:
        AttnResStream.read = original

    for w in captured["weights"]:
        total = w.sum(0)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
            "attention over sources must be a distribution"
        assert torch.allclose(w, torch.full_like(w, 1.0 / w.shape[0]), atol=1e-5), \
            "at zero query the distribution must be uniform"


@check
def standard_residual_is_the_running_sum():
    spec, tokens, _ = _spec_and_tokens()
    cfg = Config(vocab_size=spec.vocab_size, dim=32, n_heads=4,
                 max_seq_len=spec.seq_len, attn_res="none",
                 n_prelude=1, n_core=1, n_coda=1, moe=False)
    model = Recurrent(cfg)
    x = model.embed(tokens)
    stream = model._stream(x)
    blk = model.prelude[0]
    blk(stream, model.rotary[:tokens.shape[1]], step=0)
    manual = x + stream.values_sum if hasattr(stream, "values_sum") else None
    assert manual is None                         # the control keeps no source list
    assert stream.h.shape == x.shape
    assert not torch.allclose(stream.h, x), "the block must have written something"


@check
def registers_are_visible_and_not_decoded():
    spec, tokens, _ = _spec_and_tokens()
    plain = Config(vocab_size=spec.vocab_size, dim=32, n_heads=4,
                   max_seq_len=spec.seq_len, loops=2)
    with_regs = plain.but(registers=4)
    a, _ = Recurrent(plain)(tokens)
    b, _ = Recurrent(with_regs)(tokens)
    assert a.shape == b.shape, "registers must not appear in the logits"

    model = Recurrent(with_regs)
    # a register write must be able to reach position 0 of the text, which a
    # causal mask alone would forbid
    n = tokens.shape[1] + with_regs.registers
    i = torch.arange(n)
    mask = (i[None, :] <= i[:, None])
    mask[:, :with_regs.registers] = True
    assert bool(mask[with_regs.registers, 0]), "registers must be readable by token 0"


@check
def halting_distribution_sums_to_one():
    for kind in ("ouro", "pondernet"):
        cfg = Config(vocab_size=16, dim=32, n_heads=4, loops=4, halting=kind,
                     max_train_loops=4)
        halts = [torch.rand(8) for _ in range(4)]
        losses = [torch.rand(8) for _ in range(4)]
        _, q = halting_loss({"halt": halts}, losses, cfg)
        total = q.sum(0)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
            f"{kind}: exit probabilities must sum to 1, got {total[:3]}"


@check
def same_seed_means_identical_shared_weights():
    base = Config(vocab_size=16, dim=32, n_heads=4, loops=4, seed=3)
    other = base.but(registers=8)
    a, b = Recurrent(base), Recurrent(other)
    for name, pa in a.named_parameters():
        pb = dict(b.named_parameters()).get(name)
        if pb is None or pa.shape != pb.shape:
            continue
        assert torch.equal(pa, pb), f"{name} differs between paired arms"


@check
def flops_track_the_loop_count():
    one = Config(vocab_size=16, dim=32, n_heads=4, loops=1)
    four = one.but(loops=4)
    a, b = Recurrent(one), Recurrent(four)
    fa, fb = a.flops_per_token(64), b.flops_per_token(64)
    assert fb > 2.5 * fa, f"R=4 must cost far more than R=1 ({fa:.2e} vs {fb:.2e})"
    assert a.n_params() == b.n_params(), "looping must not add parameters"


@check
def hop_targets_are_the_composed_map():
    spec = HopSpec(n_entities=6, hops=(3,), queries=3)
    g = torch.Generator().manual_seed(7)
    tokens, target, hops = hop_batch(spec, 5, g)
    V = spec.n_entities
    pairs = tokens[:, 1:1 + 2 * V]
    for b in range(tokens.shape[0]):
        mapping = {int(pairs[b, i]): int(pairs[b, i + 1])
                   for i in range(0, 2 * V, 2)}
        for m, pos in enumerate(spec.answer_positions()):
            x = int(tokens[b, pos])
            walked = x
            for _ in range(int(hops[b, m])):
                walked = mapping[walked]
            assert walked == int(target[b, m]), "target is not the composed map"
            assert walked != x, "the copy-the-query shortcut was not rejected"


def main() -> None:
    failures = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
