"""Autoregressive generation, `K` tokens at a time.

The loop is the ordinary one with the unit of work enlarged: sample the next
*latent* from the head, decode it to `K` tokens with the frozen codec, append
them, repeat. A model with `patch_size = 4` therefore takes a quarter of the
sequential steps a token-level model takes for the same output length, which
is the inference half of the compression argument.

Sampling happens in two places and they are different in kind. The head's
draw is the *model's* randomness — there is no temperature on it, because
there is no density to sharpen; noise in, latent out. The codec's decode is a
softmax over the vocabulary, so `temperature` there is ordinary token
sampling, and `temperature = 0` takes the argmax.
"""

from __future__ import annotations

import torch

from .model import CalmKimiK3


@torch.no_grad()
def generate(
    model: CalmKimiK3,
    prompt_ids: torch.Tensor,
    max_new_patches: int = 8,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Continue `prompt_ids` by `max_new_patches` patches.

    Args:
        prompt_ids: `(B, T)`, `T` a multiple of `patch_size` and at least one
            patch long.
        max_new_patches: patches to append; the sequence grows by
            `max_new_patches * patch_size` tokens.
        temperature: token-level temperature for the codec's decode.
        eos_token_id: stop early once every row has emitted it.

    Returns:
        `(B, T + max_new_patches * patch_size)` token ids.
    """
    was_training = model.training
    model.eval()

    patch_size = model.patch_size
    if prompt_ids.dim() != 2:
        raise ValueError(f"expected (batch, seq), got {tuple(prompt_ids.shape)}")
    if prompt_ids.size(1) % patch_size or prompt_ids.size(1) == 0:
        raise ValueError(
            f"prompt length {prompt_ids.size(1)} must be a positive multiple "
            f"of patch_size {patch_size}"
        )

    tokens = prompt_ids
    for _ in range(max_new_patches):
        latent = model.sample_next_latent(tokens, generator=generator)
        logits = model.autoencoder.decode(latent.unsqueeze(1))

        if temperature <= 0.0:
            next_tokens = logits.argmax(-1)
        else:
            probabilities = torch.softmax(logits.float() / temperature, dim=-1)
            next_tokens = torch.multinomial(
                probabilities.reshape(-1, probabilities.size(-1)),
                num_samples=1,
                generator=generator,
            ).reshape(probabilities.shape[:-1])

        tokens = torch.cat([tokens, next_tokens], dim=1)
        if eos_token_id is not None and (tokens == eos_token_id).any(dim=1).all():
            break

    if was_training:
        model.train()
    return tokens


@torch.no_grad()
def generate_baseline(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The same loop for a baseline `KimiK3`, one token per step.

    Kept here rather than imported so the two generation paths can be timed
    against each other without the comparison depending on which convenience
    wrapper each side happens to ship.
    """
    was_training = model.training
    model.eval()

    tokens = prompt_ids
    for _ in range(max_new_tokens):
        logits = model(input_ids=tokens).logits[:, -1].float()
        if temperature <= 0.0:
            next_token = logits.argmax(-1, keepdim=True)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
        tokens = torch.cat([tokens, next_token], dim=1)

    if was_training:
        model.train()
    return tokens


__all__ = ["generate", "generate_baseline"]
