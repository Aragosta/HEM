"""Real benchmark corpora, with their published splits.

The corpus was the weakest part of this project for a long time. Experiments ran
first on a synthetic tree grammar of 64 symbols and 1024 tokens, then on the
repository's own files -- 86% Python source -- with a homemade file-level split.
Neither is a language-modelling corpus, and no number measured on them is
comparable to anything published.

This module uses **WikiText-2** and **Penn Treebank** with their **official
train/valid/test splits**. That matters for two reasons beyond size: the splits
are the ones every published result uses, so the leakage question is settled by
the dataset rather than by our own file partitioning; and byte-level results on
them are comparable in kind to the bits-per-character literature.

============  ===========  ===========  ==========  ==============================
corpus        train        valid        test        what it is
============  ===========  ===========  ==========  ==============================
wikitext2      10,797,148   1,121,681   1,256,449   Wikipedia articles, 2.05M words
ptb             5,101,618     399,782     449,945   Penn Treebank, preprocessed
============  ===========  ===========  ==========  ==============================

WikiText-2 is the small sibling of the WikiText-103 that HELM and CALM both
train on, so a tier-2 or tier-3 run scales within the same corpus family rather
than switching domains. PTB is a second domain, useful for checking that a
result is not an artefact of one corpus.

**Byte level, deliberately.** Both corpora are read as bytes (WikiText-2 uses 178
distinct byte values). Byte level removes the tokenizer from the comparison,
which is why bits-per-byte is what gets reported when models do not share a
vocabulary -- and HELM's Llama-3 tokenizer is not downloadable in this
environment anyway. A word-level mode is available for comparison against
word-level perplexity tables.

If the files are missing, :func:`load` raises with the command that fetches
them rather than silently falling back to repository text. A silent fallback is
how a study ends up reporting numbers from a corpus nobody meant to use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

DATA = Path(__file__).resolve().parents[1] / "data"

SOURCES: Dict[str, Dict[str, str]] = {
    "wikitext2": {
        "base": ("https://raw.githubusercontent.com/pytorch/examples/main/"
                 "word_language_model/data/wikitext-2"),
        "pattern": "wikitext2.{split}.txt",
    },
    "ptb": {
        "base": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data",
        "pattern": "ptb.{split}.txt",
    },
}
SPLITS = ("train", "valid", "test")


@dataclass
class Corpus:
    """Token tensors for one dataset's official splits, plus provenance."""
    name: str
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    level: str
    vocab_size: int
    digests: Dict[str, str]

    def describe(self) -> str:
        return (f"{self.name} ({self.level}-level, vocab {self.vocab_size}): "
                f"{self.train.numel():,} train / {self.valid.numel():,} valid / "
                f"{self.test.numel():,} test units, official splits")


def fetch_command(name: str) -> str:
    source = SOURCES[name]
    return " && ".join(
        f'curl -sSL -o {DATA / source["pattern"].format(split=s)} '
        f'{source["base"]}/{s}.txt' for s in SPLITS)


def _read(name: str, split: str) -> bytes:
    path = DATA / SOURCES[name]["pattern"].format(split=split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Fetch the corpus rather than falling back to "
            f"other text -- a silent fallback is how a study reports numbers "
            f"from a corpus nobody meant to use:\n  {fetch_command(name)}")
    return path.read_bytes()


def load(name: str = "wikitext2", level: str = "byte",
         limit: Optional[int] = None) -> Corpus:
    """Load a corpus with its published splits.

    Args:
        name: ``"wikitext2"`` or ``"ptb"``.
        level: ``"byte"`` (vocab 256, tokenizer-free) or ``"word"`` (vocabulary
            built on the train split; unseen validation words map to ``<unk>``,
            which both corpora already contain).
        limit: keep only the first N units of each split, for smoke tests. It
            changes what is being measured, so the runner records it.
    """
    if name not in SOURCES:
        raise ValueError(f"unknown corpus {name!r}; have {sorted(SOURCES)}")
    raw = {split: _read(name, split) for split in SPLITS}
    digests = {s: hashlib.sha256(b).hexdigest()[:16] for s, b in raw.items()}

    if level == "byte":
        tensors = {s: torch.frombuffer(bytearray(b), dtype=torch.uint8).long()
                   for s, b in raw.items()}
        vocab = 256
    elif level == "word":
        words = {s: b.decode("utf-8", "replace").split() for s, b in raw.items()}
        vocabulary = {token: i for i, token in enumerate(sorted(set(words["train"])))}
        unk = vocabulary.get("<unk>", 0)
        tensors = {s: torch.tensor([vocabulary.get(t, unk) for t in seq],
                                   dtype=torch.long)
                   for s, seq in words.items()}
        vocab = len(vocabulary)
    else:
        raise ValueError(f"level must be 'byte' or 'word', got {level!r}")

    if limit:
        tensors = {s: t[:limit] for s, t in tensors.items()}
    return Corpus(name=name, level=level, vocab_size=vocab, digests=digests,
                  **{s: tensors[s] for s in SPLITS})


def batches_from(data: torch.Tensor, batch_size: int, seq_len: int, count: int,
                 seed: int) -> List[torch.Tensor]:
    """Fixed random windows, so every cell of the 2x2 sees identical data."""
    generator = torch.Generator().manual_seed(seed)
    highest = data.numel() - seq_len - 1
    if highest <= 0:
        raise ValueError(f"split of {data.numel()} units is shorter than "
                         f"seq_len {seq_len}")
    out = []
    for _ in range(count):
        starts = torch.randint(0, highest, (batch_size,), generator=generator)
        out.append(torch.stack([data[s:s + seq_len] for s in starts]))
    return out


def overlap_report(corpus: Corpus, n: int = 16, samples: int = 20000,
                   seed: int = 0) -> Dict[str, float]:
    """How much of the validation split appears verbatim in train.

    Published splits are built to avoid leakage, but the check is cheap and it
    is exactly what the earlier homemade file-level split needed and never had.
    A high figure means held-out accuracy is partly recall.
    """
    train_list = corpus.train.tolist()
    step = max(len(train_list) // 400_000, 1)
    train_grams = {tuple(train_list[i:i + n])
                   for i in range(0, len(train_list) - n, step)}
    valid_list = corpus.valid.tolist()
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randint(0, max(len(valid_list) - n, 1), (samples,),
                              generator=generator).tolist()
    hits = sum(tuple(valid_list[p:p + n]) in train_grams for p in positions)
    return {"n": float(n), "sampled": float(samples),
            "verbatim_fraction": hits / samples}


if __name__ == "__main__":
    for corpus_name in SOURCES:
        try:
            data = load(corpus_name)
        except FileNotFoundError as error:
            print(error)
            continue
        print(data.describe())
        print(f"  digests {data.digests}")
        print(f"  16-gram verbatim overlap valid->train: "
              f"{overlap_report(data)['verbatim_fraction']:.3%}")
