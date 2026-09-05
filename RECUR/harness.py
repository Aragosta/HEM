"""Training, evaluation and the bookkeeping that makes two arms comparable.

Three things here are not incidental and are the reason results from this
folder can be read at all:

**Pairing.** Arms that differ in a config field share the model init seed *and*
the data stream. On the composition tasks the evaluation set is generated once
from its own seed and reused by every arm, so a difference between two arms is
a difference in the arms, not in what they were shown or scored on. Every
result file records both seeds.

**A noise floor before any claim.** :func:`seed_spread` runs one config at
several seeds. No difference smaller than that spread is reported as a
difference; ``e0_baseline.py`` measures it first and every later experiment
quotes it.

**Compute, not parameters.** A looped model reuses its core, so it is cheap in
parameters and expensive in FLOPs. Every run records parameters, active
parameters and forward FLOPs per token, and the fixed-compute experiments
match on the last of those. Matching on parameters alone is how a looped model
gets to look free.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from model import Config, Recurrent
from tasks import HopSpec, hop_batch, hop_eval_set, byte_batches, load_bytes

RESULTS = Path(__file__).resolve().parent / "results"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


# ------------------------------------------------------------------ losses

def halting_loss(aux: Dict, per_step_loss: List[torch.Tensor], cfg: Config):
    """Expected loss over exit steps, plus the prior regulariser.

    ``ouro``: a step-indexed gate and an entropy regulariser against a uniform
    prior over the ``T`` trained depths -- Ouro's formulation, reproduced
    including the part that cannot survive unbounded depth.

    ``pondernet``: a step-*invariant* gate (one head applied to the state at
    every step, no dependence on the step index) with a geometric prior. This
    is the fix proposed in the brief; the point of running both is that the
    fix is only worth its hyperparameter if it buys depth extrapolation.
    """
    halts = aux["halt"]
    T = len(halts)
    remain = torch.ones_like(halts[0])
    q, expected = [], 0.0
    for t in range(T):
        lam = halts[t] if t < T - 1 else torch.ones_like(halts[t])
        q_t = remain * lam
        q.append(q_t)
        expected = expected + (q_t * per_step_loss[t]).mean()
        remain = remain * (1 - lam)
    qs = torch.stack(q, 0).clamp_min(1e-8)               # (T, B)

    if cfg.halting == "ouro":
        prior = torch.full_like(qs, 1.0 / T)
    else:
        lam_p = cfg.halt_prior
        w = torch.tensor([lam_p * (1 - lam_p) ** t for t in range(T)],
                         device=qs.device)
        prior = (w / w.sum()).unsqueeze(1).expand_as(qs)
    kl = (qs * (qs.log() - prior.log())).sum(0).mean()
    return expected + cfg.halt_beta * kl, qs.detach()


# ------------------------------------------------------------------- runners

def _optimizer(model, lr, weight_decay=0.01):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))


def _lr_at(step, total, lr, warmup=0.05):
    w = max(1, int(total * warmup))
    if step < w:
        return lr * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def answer_logits(logits: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    """(B, M, V): the prediction made at each question's last context token."""
    return logits[:, list(positions), :]


@torch.no_grad()
def evaluate_hops(model, eval_set, spec, loops: Optional[int] = None,
                  collect: bool = False) -> Dict:
    model.eval()
    positions = spec.answer_positions()
    out = {}
    for hop, (tokens, target, _) in eval_set.items():
        logits, aux = model(tokens, loops=loops, collect=collect)
        pred = answer_logits(logits, positions)
        out[f"acc_h{hop}"] = (pred.argmax(-1) == target).float().mean().item()
        out[f"loss_h{hop}"] = F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), target.reshape(-1)).item()
        if collect and aux.get("step_logits"):
            for r, sl in enumerate(aux["step_logits"]):
                out[f"acc_h{hop}_r{r + 1}"] = (
                    answer_logits(sl, positions).argmax(-1)
                    == target).float().mean().item()
    accs = [v for k, v in out.items() if k.startswith("acc_h") and "_r" not in k]
    out["acc"] = sum(accs) / len(accs)
    model.train()
    return out


@torch.no_grad()
def evaluate_bytes(model, batches, loops: Optional[int] = None) -> Dict:
    model.eval()
    total, count = 0.0, 0
    for batch in batches:
        logits, _ = model(batch[:, :-1], loops=loops)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               batch[:, 1:].reshape(-1))
        total += loss.item() * batch[:, 1:].numel()
        count += batch[:, 1:].numel()
    model.train()
    nats = total / count
    return {"loss": nats, "bpb": nats / math.log(2)}


@torch.no_grad()
def evaluate_halting(model, eval_set, spec, loops: int, q: float = 0.5) -> Dict:
    """Q-exit: stop at the first step whose cumulative exit CDF passes ``q``.

    Reports the accuracy that rule achieves and the average depth it spends, so
    a gate is scored on the tradeoff it was trained to make rather than on
    accuracy at a fixed depth.
    """
    model.eval()
    positions = spec.answer_positions()
    out = {}
    for hop, (tokens, target, _) in eval_set.items():
        _, aux = model(tokens, loops=loops, collect=True)
        halts = aux["halt"]
        if not halts:
            continue
        remain = torch.ones_like(halts[0])
        cdf = torch.zeros_like(halts[0])
        chosen = torch.full_like(halts[0], float(loops))
        correct = torch.zeros_like(halts[0])
        done = torch.zeros_like(halts[0], dtype=torch.bool)
        for t, lam in enumerate(halts):
            lam_t = lam if t < len(halts) - 1 else torch.ones_like(lam)
            cdf = cdf + remain * lam_t
            remain = remain * (1 - lam_t)
            hit = (~done) & (cdf >= q)
            step_acc = (answer_logits(aux["step_logits"][t], positions).argmax(-1)
                        == target).float().mean(1)
            correct = torch.where(hit, step_acc, correct)
            chosen = torch.where(hit, torch.full_like(chosen, t + 1.0), chosen)
            done = done | hit
        out[f"qexit_acc_h{hop}"] = correct.mean().item()
        out[f"qexit_depth_h{hop}"] = chosen.mean().item()
    model.train()
    return out


@torch.no_grad()
def evaluate_kl_exit(model, eval_set, spec, loops: int,
                     taus=(0.5, 0.1, 0.02)) -> Dict:
    """Huginn's zero-shot exit rule, on a model trained without a gate.

    Exit at the first step where the KL between successive next-token
    distributions falls below ``tau``. Costs no training, which is exactly the
    point of including it next to the two learned gates.
    """
    model.eval()
    positions = spec.answer_positions()
    out = {}
    for hop, (tokens, target, _) in eval_set.items():
        _, aux = model(tokens, loops=loops, collect=True)
        steps = [answer_logits(sl, positions).log_softmax(-1)
                 for sl in aux["step_logits"]]
        for tau in taus:
            depth = torch.full(steps[0].shape[:2], float(loops))
            correct = (steps[-1].argmax(-1) == target).float()
            done = torch.zeros_like(depth, dtype=torch.bool)
            for t in range(1, len(steps)):
                kl = (steps[t].exp() * (steps[t] - steps[t - 1])).sum(-1)
                hit = (~done) & (kl < tau)
                correct = torch.where(hit, (steps[t].argmax(-1) == target).float(),
                                      correct)
                depth = torch.where(hit, torch.full_like(depth, t + 1.0), depth)
                done = done | hit
            out[f"kl{tau}_acc_h{hop}"] = correct.mean().item()
            out[f"kl{tau}_depth_h{hop}"] = depth.mean().item()
    model.train()
    return out


@torch.no_grad()
def routing_histograms(model, tokens) -> List[List[float]]:
    """Expert usage per loop, so "does routing actually differ per step" is
    measurable rather than assumed.

    A collected forward returns one index tensor per (loop, MoE block) in
    order; they are summed within a loop and normalised, giving one
    distribution over experts per loop.
    """
    cfg = model.cfg
    if not cfg.moe:
        return []
    model.eval()
    _, aux = model(tokens, collect=True)
    model.train()
    idxs = aux.get("routing", [])
    per_loop = []
    for start in range(0, len(idxs), cfg.n_core):
        counts = torch.zeros(cfg.n_routed)
        for t in idxs[start:start + cfg.n_core]:
            counts += torch.bincount(t.reshape(-1), minlength=cfg.n_routed).float()
        per_loop.append((counts / counts.sum().clamp_min(1)).tolist())
    return per_loop


def train_hops(cfg: Config, spec: HopSpec, steps: int, batch_size: int = 64,
               lr: float = 3e-3, data_seed: int = 1234, eval_seed: int = 99,
               eval_size: int = 256, eval_every: int = 0,
               eval_loops: Sequence[int] = (),
               train_loops: Optional[Callable[[int], int]] = None) -> Dict:
    """Train on fresh in-context graphs; score on a fixed evaluation set.

    ``train_loops`` lets an arm sample its recurrence count per batch (Huginn's
    log-normal-Poisson schedule lives in :func:`sampled_loops`); the evaluation
    always runs at the configured depth unless asked otherwise.
    """
    model = Recurrent(cfg)
    opt = _optimizer(model, lr)
    g = torch.Generator().manual_seed(data_seed)
    eval_set = hop_eval_set(spec, eval_size, eval_seed)
    positions = spec.answer_positions()
    history, t0 = [], time.time()

    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, steps, lr)
        tokens, target, _ = hop_batch(spec, batch_size, g)
        loops = train_loops(step) if train_loops else None
        logits, aux = model(tokens, loops=loops)
        flat_target = target.reshape(-1)
        if cfg.halting != "none":
            per_step = [F.cross_entropy(
                answer_logits(sl, positions).reshape(-1, sl.shape[-1]),
                flat_target, reduction="none").view(target.shape).mean(1)
                for sl in aux["step_logits"]]
            loss, _ = halting_loss(aux, per_step, cfg)
        else:
            pred = answer_logits(logits, positions)
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), flat_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.rebalance()
        if eval_every and (step + 1) % eval_every == 0:
            history.append({"step": step + 1, "train_loss": loss.item(),
                            **evaluate_hops(model, eval_set, spec)})

    final = evaluate_hops(model, eval_set, spec, collect=True)
    by_depth = {str(r): evaluate_hops(model, eval_set, spec, loops=r)
                for r in eval_loops}
    exits = {}
    if cfg.halting != "none":
        for r in set([cfg.loops, *eval_loops]):
            exits[str(r)] = evaluate_halting(model, eval_set, spec, loops=r)
    else:
        exits[str(cfg.loops)] = evaluate_kl_exit(model, eval_set, spec,
                                                 loops=cfg.loops)
    return {
        "config": asdict(cfg), "task": "twochain" if spec.two_chain else "hops",
        "spec": asdict(spec), "steps": steps, "batch_size": batch_size, "lr": lr,
        "data_seed": data_seed, "eval_seed": eval_seed,
        "params": model.n_params(), "active_params": model.n_active_params(),
        "flops_per_token": model.flops_per_token(spec.seq_len),
        "train_tokens": steps * batch_size * spec.seq_len,
        "train_answers": steps * batch_size * spec.queries,
        "train_flops": 3 * model.flops_per_token(spec.seq_len) * steps * batch_size * spec.seq_len,
        "seconds": time.time() - t0, "history": history, "final": final,
        "by_depth": by_depth, "exits": exits,
        "routing_hist": routing_histograms(model, eval_set[spec.hops[0]][0][:64]),
        "expert_load": [l.tolist() for l in model.expert_load()],
        "commit": git_commit(),
    }, model


def train_bytes(cfg: Config, corpus, steps: int, seq_len: int = 128,
                batch_size: int = 16, lr: float = 1.5e-3, data_seed: int = 1234,
                eval_batches: int = 24) -> Dict:
    model = Recurrent(cfg)
    opt = _optimizer(model, lr)
    train = byte_batches(corpus.splits["train"], batch_size, seq_len, steps, data_seed)
    valid = byte_batches(corpus.splits["valid"], batch_size, seq_len, eval_batches, 7)
    t0 = time.time()
    for step, batch in enumerate(train):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, steps, lr)
        logits, aux = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               batch[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.rebalance()
    final = evaluate_bytes(model, valid)
    hist = routing_histograms(model, valid[0][:, :-1])
    return {
        "config": asdict(cfg), "task": f"bytes:{corpus.name}", "steps": steps,
        "seq_len": seq_len, "batch_size": batch_size, "lr": lr,
        "data_seed": data_seed, "digests": corpus.digests,
        "params": model.n_params(), "active_params": model.n_active_params(),
        "flops_per_token": model.flops_per_token(seq_len),
        "train_tokens": steps * batch_size * seq_len,
        "train_flops": 3 * model.flops_per_token(seq_len) * steps * batch_size * seq_len,
        "seconds": time.time() - t0, "final": final, "routing_hist": hist,
        "expert_load": [l.tolist() for l in model.expert_load()],
        "commit": git_commit(),
    }, model


def sampled_loops(mean_loops: int, max_loops: int, seed: int = 0):
    """Huginn's per-batch recurrence sampling, truncated to ``max_loops``.

    A log-normal Poisson in the paper; a clipped Poisson here, which has the
    property that matters -- the model sees a range of depths rather than one,
    which is what lets it be unrolled to depths it was not trained at.
    """
    g = torch.Generator().manual_seed(seed)

    def pick(step: int) -> int:
        lam = torch.tensor(float(mean_loops))
        r = int(torch.poisson(lam.unsqueeze(0), generator=g).item())
        return max(1, min(max_loops, r))
    return pick


# ---------------------------------------------------------------- utilities

def seed_spread(values: Sequence[float]) -> Dict[str, float]:
    if len(values) < 2:
        return {"mean": float(values[0]), "sd": float("nan"), "n": len(values)}
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "n": len(values), "min": min(values), "max": max(values)}


def save(name: str, payload: Dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
