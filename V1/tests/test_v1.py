"""Tests for the continuous-autoregressive path.

Run: `python -m pytest V1/tests -q`, or `python V1/tests/test_v1.py` to get
the same checks without pytest.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from V1.autoencoder import PatchAutoencoder  # noqa: E402
from V1.config import (  # noqa: E402
    EnergyHeadConfig,
    PatchAutoencoderConfig,
    calm_k3_cpu_tiny_config,
)
from V1.data import SyntheticCorpus  # noqa: E402
from V1.energy_head import EnergyHead, energy_score  # noqa: E402
from V1.generation import generate  # noqa: E402
from V1.metrics import brier_lm, brier_scores, self_test  # noqa: E402
from V1.model import CalmKimiK3  # noqa: E402


def _tiny_model(seed: int = 0) -> CalmKimiK3:
    torch.manual_seed(seed)
    return CalmKimiK3(calm_k3_cpu_tiny_config())


# ------------------------------------------------------------------ codec


def test_codec_shapes_and_round_trip():
    config = PatchAutoencoderConfig(vocab_size=32, patch_size=4, hidden_size=32,
                                    latent_size=8)
    codec = PatchAutoencoder(config)
    ids = torch.randint(0, 32, (3, 16))

    mean, log_std = codec.encode(ids)
    assert mean.shape == (3, 4, 8), mean.shape
    assert log_std.shape == (3, 4, 8)
    assert torch.isfinite(log_std).all()

    logits = codec.decode(mean)
    assert logits.shape == (3, 16, 32), logits.shape

    output = codec(ids)
    assert output.loss.requires_grad
    assert output.kl_loss >= 0.0


def test_codec_rejects_ragged_sequences():
    codec = PatchAutoencoder(
        PatchAutoencoderConfig(vocab_size=16, patch_size=4, hidden_size=16,
                               latent_size=4)
    )
    try:
        codec.encode(torch.randint(0, 16, (1, 10)))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected a ValueError on a ragged sequence")


def test_codec_output_projection_is_tied():
    codec = PatchAutoencoder(
        PatchAutoencoderConfig(vocab_size=16, patch_size=2, hidden_size=16,
                               latent_size=4)
    )
    assert codec.decoder.output_weight is codec.encoder.embed_tokens.weight


# ------------------------------------------------------------ energy score


def test_energy_score_prefers_the_true_distribution():
    """A strictly proper rule is maximised by matching the target."""
    torch.manual_seed(0)
    mean = torch.zeros(64, 4)
    log_std = torch.zeros(64, 4)

    matched = torch.randn(32, 64, 4)                      # same distribution
    biased = torch.randn(32, 64, 4) + 3.0                 # shifted
    collapsed = torch.zeros(32, 64, 4)                    # no spread at all

    good = energy_score(matched, mean, log_std, num_target_samples=64).mean()
    shifted = energy_score(biased, mean, log_std, num_target_samples=64).mean()
    point = energy_score(collapsed, mean, log_std, num_target_samples=64).mean()

    assert good > shifted, (good, shifted)
    assert good > point, (good, point)


def test_energy_score_rejects_a_single_sample():
    try:
        energy_score(torch.zeros(1, 2, 3), torch.zeros(2, 3), torch.zeros(2, 3))
    except ValueError as error:
        assert ">= 2" in str(error)
    else:
        raise AssertionError("expected a ValueError for one sample")


def test_energy_head_starts_as_the_zero_map():
    """Zero init means identical samples, so the self term starts at zero."""
    head = EnergyHead(EnergyHeadConfig(hidden_size=16, latent_size=8, noise_size=8))
    hidden = torch.randn(2, 5, 16)
    samples = head.sample(hidden, num_samples=4)
    assert samples.shape == (4, 2, 5, 8)
    assert torch.allclose(samples, torch.zeros_like(samples), atol=1e-6)


def test_energy_head_samples_differ_once_trained():
    """After a single gradient step the head is no longer degenerate."""
    torch.manual_seed(0)
    head = EnergyHead(EnergyHeadConfig(hidden_size=16, latent_size=8, noise_size=8))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.5)
    hidden = torch.randn(4, 16)
    mean, log_std = torch.randn(4, 8), torch.zeros(4, 8)
    for _ in range(5):
        samples = head.sample(hidden, num_samples=8)
        loss = -energy_score(samples, mean, log_std, num_target_samples=16).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    samples = head.sample(hidden, num_samples=8)
    assert samples.std(dim=0).mean() > 0.0


# ------------------------------------------------------------------ model


def test_model_forward_shapes_and_backward():
    model = _tiny_model()
    config = model.config
    ids = torch.randint(0, config.vocab_size, (2, 32))
    output = model(ids)

    patches = 32 // config.patch_size
    assert output.hidden_states.shape == (2, patches - 1, config.d_model)
    assert output.samples.shape == (
        config.energy_head.num_samples, 2, patches - 1,
        config.autoencoder.latent_size,
    )
    assert torch.isfinite(output.loss)

    output.loss.backward()
    missing = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, missing


def test_codec_is_frozen_and_stays_in_eval():
    model = _tiny_model()
    assert not any(p.requires_grad for p in model.autoencoder.parameters())
    model.train()
    assert not model.autoencoder.training, "a frozen codec must not re-enter train mode"

    ids = torch.randint(0, model.config.vocab_size, (2, 32))
    before = model.autoencoder.encoder.embed_tokens.weight.clone()
    model(ids).loss.backward()
    assert torch.equal(before, model.autoencoder.encoder.embed_tokens.weight)


def test_backbone_sees_compressed_sequence():
    """The point of the method: T tokens become T/K backbone positions."""
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (2, 64))
    embeddings = model.encode_patches(ids)
    assert embeddings.shape == (2, 64 // model.patch_size, model.config.d_model)


def test_model_rejects_short_and_ragged_input():
    model = _tiny_model()
    for bad in (torch.randint(0, 8, (1, 6)), torch.randint(0, 8, (1, 4))):
        try:
            model(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected a ValueError for shape {tuple(bad.shape)}")


def test_loss_decreases_on_a_fixed_batch():
    """The objective is optimisable, not merely finite."""
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (4, 32))
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)

    generator = torch.Generator().manual_seed(1234)
    first = float(model(ids, generator=generator).loss)
    for _ in range(30):
        loss = model(ids).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    generator = torch.Generator().manual_seed(1234)
    last = float(model(ids, generator=generator).loss)
    assert last < first, (first, last)


# ------------------------------------------------------------- generation


def test_generate_extends_by_whole_patches():
    model = _tiny_model()
    prompt = torch.randint(0, model.config.vocab_size, (2, 8))
    out = generate(model, prompt, max_new_patches=3)
    assert out.shape == (2, 8 + 3 * model.patch_size)
    assert torch.equal(out[:, :8], prompt)
    assert out.max() < model.config.vocab_size and out.min() >= 0


def test_generate_rejects_a_ragged_prompt():
    model = _tiny_model()
    try:
        generate(model, torch.randint(0, 8, (1, 5)), max_new_patches=1)
    except ValueError as error:
        assert "multiple" in str(error)
    else:
        raise AssertionError("expected a ValueError on a ragged prompt")


def test_predict_tokens_shape():
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (2, 32))
    tokens = model.predict_tokens(ids)
    assert tokens.shape == (2, 32 // model.patch_size - 1, model.patch_size)


# ---------------------------------------------------------------- metrics


def test_brier_estimator_anchors():
    self_test()


def test_brier_penalises_collapse():
    targets = torch.randint(0, 50, (32, 8))
    collapsed = torch.zeros_like(targets)
    assert brier_scores(collapsed, collapsed, targets)[0] < 0
    assert brier_lm(targets, targets, targets) == 1.0


# ------------------------------------------------------------------- data


def test_corpus_is_deterministic_and_structured():
    corpus = SyntheticCorpus(vocab_size=64)
    a = corpus.sample(4, 32, seed=7)
    b = corpus.sample(4, 32, seed=7)
    assert torch.equal(a, b)
    assert not torch.equal(a, corpus.sample(4, 32, seed=8))
    # Structure means the conditional entropy is well below uniform.
    assert corpus.markov_entropy_floor() < math.log(64)


def _run_all() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
