"""The batched scorer must agree with the one-request-at-a-time original."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.eval.presets import PRESETS, preset_args  # noqa: E402
from helm.eval.scoring import (generate, rolling_logprob,  # noqa: E402
                               score_continuations)
from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from tests._config import tiny_args  # noqa: E402


def upstream_score(model, prompt_ids, cont_ids, max_seq_len=2048):
    """The released ``_score_sequence``, transcribed, as the reference."""
    prompt_ids = torch.tensor([list(prompt_ids)])
    cont_ids = torch.tensor([list(cont_ids)])
    cont_len = cont_ids.size(1)
    prompt_ids = prompt_ids[..., -(max_seq_len - cont_len):]
    prompt_len = prompt_ids.size(1)
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)

    with torch.no_grad():
        logits = model(input_ids)

    total_logprob = 0.0
    greedy = True
    for i in range(cont_len):
        next_logit = logits[0, prompt_len + i - 1]
        log_probs = F.log_softmax(next_logit, dim=-1)
        total_logprob += float(log_probs[cont_ids[0, i]])
        if greedy and log_probs.argmax().item() != cont_ids[0, i]:
            greedy = False
    return total_logprob, greedy


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    args = tiny_args(max_seq_len=64, original_seq_len=64)
    return HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)).double().eval()


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_batched_scoring_matches_one_at_a_time(model, batch_size):
    """Batching and padding must not change any score."""
    rng = np.random.default_rng(0)
    vocab = tiny_args().vocab_size
    pairs = []
    for _ in range(7):
        # Deliberately ragged, so batches pad by different amounts.
        n_ctx = int(rng.integers(3, 20))
        n_cont = int(rng.integers(1, 6))
        pairs.append((list(rng.integers(0, vocab, n_ctx)),
                      list(rng.integers(0, vocab, n_cont))))

    got = score_continuations(model, pairs, batch_size=batch_size, max_seq_len=64,
                              pad_id=0)
    assert len(got) == len(pairs)
    for scored, (ctx, cont) in zip(got, pairs):
        want_lp, want_greedy = upstream_score(model, ctx, cont, max_seq_len=64)
        assert scored.logprob == pytest.approx(want_lp, abs=1e-8)
        assert scored.is_greedy == want_greedy


def test_scoring_preserves_request_order(model):
    """Results come back in request order despite internal length sorting."""
    pairs = [([1, 2, 3] * i, [4, 5]) for i in range(1, 6)]
    batched = score_continuations(model, pairs, batch_size=2, max_seq_len=64)
    one_by_one = [score_continuations(model, [p], batch_size=1, max_seq_len=64)[0]
                  for p in pairs]
    for a, b in zip(batched, one_by_one):
        assert a.logprob == pytest.approx(b.logprob, abs=1e-9)


def test_scoring_left_truncates_context_not_continuation(model):
    """An over-long prompt loses context from the left; the continuation stays."""
    cont = [7, 8, 9]
    vocab = tiny_args().vocab_size
    pairs = [([i % vocab for i in range(1, 200)], cont)]
    scored = score_continuations(model, pairs, batch_size=1, max_seq_len=32)
    assert len(scored) == 1 and np.isfinite(scored[0].logprob)


def test_empty_continuation_is_rejected(model):
    with pytest.raises(ValueError, match="empty continuation"):
        score_continuations(model, [([1, 2], [])], max_seq_len=64)


def test_rolling_logprob_matches_direct_scoring(model):
    """Rolling log-prob equals a single full-sequence scoring when it fits."""
    tokens = [3, 14, 15, 92, 65, 35, 89, 79]
    got = rolling_logprob(model, tokens, max_seq_len=64)
    with torch.no_grad():
        logits = model(torch.tensor([tokens]))
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = torch.tensor(tokens[1:])
    want = float(log_probs[0, :-1].gather(-1, targets[:, None]).sum())
    assert got == pytest.approx(want, abs=1e-6)


def test_rolling_logprob_windows_long_sequences(model):
    """Longer than the context window: still finite, still every token once."""
    tokens = list(range(1, 90))
    assert np.isfinite(rolling_logprob(model, tokens, max_seq_len=32))


def test_generate_matches_uncached_greedy_decoding(model):
    """Cached greedy generation equals re-running the full prefix each step."""
    prompt = [5, 6, 7, 8]
    got = generate(model, prompt, max_new_tokens=6, temperature=0.0)

    want = []
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(6):
            logits = model(torch.tensor([tokens]))
            nxt = int(logits[0, -1].argmax())
            want.append(nxt)
            tokens.append(nxt)
    assert got == want


def test_generate_stops_on_stop_token(model):
    produced = generate(model, [5, 6], max_new_tokens=10, temperature=0.0)
    stopped = generate(model, [5, 6], max_new_tokens=10, temperature=0.0,
                       stop_ids=[produced[0]])
    assert stopped == produced[:1]


def test_presets_match_released_shapes():
    """Guard the transcribed configs against typos."""
    p120 = preset_args("helm_mice_120M")
    assert (p120.dim, p120.n_layers, p120.n_heads, p120.n_routed_experts) == (390, 6, 6, 4)
    assert (p120.kv_lora_rank, p120.qk_nope_head_dim, p120.qk_rope_head_dim,
            p120.v_head_dim) == (65, 33, 17, 33)
    p1b = preset_args("helm_mice_1B")
    assert (p1b.dim, p1b.n_layers, p1b.n_heads, p1b.n_routed_experts) == (896, 15, 14, 8)
    assert p1b.project_emb is True
    assert set(PRESETS) == {"helm_mice_120M", "helm_mice_1B", "helm_d_115M"}


def test_preset_builds_a_working_model():
    """A released preset must actually construct and run."""
    args = preset_args("helm_mice_120M", max_seq_len=16, original_seq_len=16,
                       vocab_size=64, n_layers=2)
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)).eval()
    with torch.no_grad():
        logits = model(torch.randint(0, 64, (1, 8)))
    assert logits.shape == (1, 8, 64)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
