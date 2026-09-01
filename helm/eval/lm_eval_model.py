"""``lm-evaluation-harness`` integration for HELM models.

Importing this module registers ``helm_mice_120M``, ``helm_mice_1B`` and
``helm_d_115M`` with lm-eval::

    lm_eval --model helm_mice_120M \\
            --model_args ckpt_dir=/path/to/Step1000.pt,batch_size=16 \\
            --tasks hellaswag --num_fewshot 0

The authors ship this as a patch inside a vendored 47 MB fork of the harness
(``lm-evaluation-harness/lm_eval/models/helm.py`` plus a second, drifted copy of
the model code under ``helm_module/``). Carrying a fork means the model code
exists twice and the copies diverge -- theirs already have: the eval copy of
``LorentzMoE`` contains the eval-mode fix that the library copy is missing, and
the 1B eval config disagrees with ``example/train_mice_1B.sh`` on three fields.

This version is a plugin against whatever lm-eval you have installed, so there
is one copy of the model, and it fixes the three things that made evaluation the
slowest part of using HELM:

* **Requests are batched.** The original scores one continuation per forward
  pass and ignores ``batch_size`` entirely.
* **Log-probabilities are gathered vectorised**, not in a Python loop that calls
  ``float()`` once per continuation token.
* **``generate_until`` works**, using the KV cache. The original raises
  ``NotImplementedError``, which excludes every generative task.

``loglikelihood_rolling`` also works here; the original cannot run at all (it
unpacks three values from a tensor slice).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch

from helm.eval.presets import preset_args
from helm.eval.scoring import generate, rolling_logprob, score_continuations

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    _HAS_LM_EVAL = True
except ImportError:  # pragma: no cover - exercised only without lm-eval
    _HAS_LM_EVAL = False
    LM = object

    def register_model(*names):
        return lambda cls: cls


DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B"


class _HelmLM(LM):
    """Shared lm-eval adapter; subclasses only pick a preset."""

    preset: str = "helm_mice_120M"

    def __init__(self, ckpt_dir: str, device: str = "cuda:0", batch_size: int = 8,
                 tokenizer: str = DEFAULT_TOKENIZER, dtype: str = "float32",
                 max_seq_len: int = 2048, **kwargs) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        from helm.hypercore.manifolds import Lorentz

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_seq_len = int(max_seq_len)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, token=os.environ.get("HF_TOKEN"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        args = preset_args(self.preset, max_seq_len=self.max_seq_len)
        manifolds = (Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
        if args.model_name == "HELM_MiCE":
            from helm.modules.helm_mice import HelmMiCE
            model = HelmMiCE(args, *manifolds)
        else:
            from helm.modules.helm_d import LTransformerDecoder
            model = LTransformerDecoder(*manifolds, args.arch, args.vocab_size,
                                        args.max_seq_len)

        state = torch.load(ckpt_dir, map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"checkpoint is missing {len(missing)} keys, "
                               f"first few: {missing[:5]}")
        self.model = model.to(device=self.device, dtype=getattr(torch, dtype)).eval()

    def _encode(self, text: str, add_special_tokens: bool) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def loglikelihood(self, requests: List["Instance"]) -> List[Tuple[float, bool]]:
        pairs = []
        for context, continuation in (request.args for request in requests):
            pairs.append((self._encode(context, True),
                          self._encode(continuation, False)))
        scored = score_continuations(
            self.model, pairs, batch_size=self.batch_size,
            max_seq_len=self.max_seq_len, pad_id=self.pad_id,
            device=self.device, progress=True)
        return [s.as_tuple() for s in scored]

    def loglikelihood_rolling(self, requests: List["Instance"]) -> List[float]:
        return [rolling_logprob(self.model, self._encode(request.args[0], True),
                                max_seq_len=self.max_seq_len, device=self.device)
                for request in requests]

    def generate_until(self, requests: List["Instance"]) -> List[str]:
        outputs = []
        for request in requests:
            context, config = request.args
            config = config or {}
            stop_strings = config.get("until", []) or []
            produced = generate(
                self.model, self._encode(context, True),
                max_new_tokens=config.get("max_gen_toks", 64),
                temperature=config.get("temperature", 0.0),
                device=self.device)
            text = self.tokenizer.decode(produced)
            for stop in stop_strings:
                if stop and stop in text:
                    text = text.split(stop)[0]
            outputs.append(text)
        return outputs


@register_model("helm_mice_120M")
class HelmMiCE120M(_HelmLM):
    preset = "helm_mice_120M"


@register_model("helm_mice_1B")
class HelmMiCE1B(_HelmLM):
    preset = "helm_mice_1B"


@register_model("helm_d_115M")
class HelmD115M(_HelmLM):
    preset = "helm_d_115M"
