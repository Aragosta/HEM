"""A BPE tokenizer trained on the corpus itself, since HELM's is not fetchable.

HELM uses the LLaMA-3.1 tokenizer (128K vocab). HuggingFace is blocked here, so
that exact vocabulary is unavailable. Byte level was the fallback used earlier
and it is the wrong setting for testing HELM specifically: **256 byte values have
no long tail**, and HELM's own case study locates the geometry's contribution in
the tail -- generic words at small norm, specific words at large norm.

Training BPE locally restores what matters: a Zipfian subword vocabulary with
rare units. Fitted on the **train split only**, so the held-out splits stay
held out.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parents[1] / "data"


def train_or_load(corpus: str = "wikitext2", vocab_size: int = 16000) -> Tokenizer:
    path = DATA / f"bpe_{corpus}_{vocab_size}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=["<unk>", "<eos>"],
                                  show_progress=False)
    tokenizer.train([str(DATA / f"{corpus}.train.txt")], trainer)
    tokenizer.save(str(path))
    return tokenizer


def encode_split(tokenizer: Tokenizer, corpus: str, split: str) -> torch.Tensor:
    text = (DATA / f"{corpus}.{split}.txt").read_text(encoding="utf-8", errors="replace")
    return torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)


if __name__ == "__main__":
    import collections
    tok = train_or_load()
    print("vocab", tok.get_vocab_size())
    for split in ("train", "valid", "test"):
        ids = encode_split(tok, "wikitext2", split)
        print(f"  {split}: {ids.numel():,} tokens")
        if split == "train":
            counts = collections.Counter(ids.tolist())
            freqs = sorted(counts.values(), reverse=True)
            print(f"    top-10 = {sum(freqs[:10])/sum(freqs):.1%} of corpus; "
                  f"units seen <5 times: {sum(1 for f in freqs if f < 5)}; "
                  f"distinct used: {len(freqs)}")
