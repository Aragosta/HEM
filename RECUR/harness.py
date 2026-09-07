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
from dataclasses import asdict, replace
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
    """AdamW, with the temperature exempt from weight decay.

    ``log_beta`` is 2-D, so the usual "decay matrices, spare vectors" rule
    would decay it -- and decaying a log-parameter pulls it toward 0, i.e. the
    multiplier toward 1.0. That is not regularisation, it is the optimiser
    quietly holding the temperature at the baseline and answering the question
    the experiment is asking. (The same trap is recorded in
    ``CALM/CRITICALITY.md``, found the same way.)
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.endswith("log_beta") else decay).append(p)
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
@torch.no_grad()
def beta_transfer(model, eval_set, spec, scales=(0.25, 0.5, 1.0, 2.0, 4.0),
                  loops: Optional[int] = None) -> Dict:
    """Train at one temperature, evaluate at several. Costs no training.

    Two literatures point opposite ways and both are about beta. Velickovic et
    al. (arXiv:2410.01104) show softmax *disperses* out of distribution and
    propose raising sharpness **at inference**; our A1 sweep says a **lower**
    beta is better **during training**. Those are compatible if the right
    schedule is soft-to-learn, sharp-to-decide -- and that is checkable for
    free, because the temperature is a multiplier on the logits and needs no
    retraining to change.

    Reports accuracy at each evaluation temperature for a model trained at one,
    so the diagonal is the usual "train and test at the same beta" number and
    the off-diagonal is the transfer.
    """
    from model import Attention
    heads = [m for m in model.modules() if isinstance(m, Attention)]
    original = [h.cfg.beta_scale for h in heads]
    positions = spec.answer_positions()
    out = {}
    model.eval()
    try:
        for scale in scales:
            for h in heads:
                h.cfg = replace(h.cfg, beta_scale=scale)
            accs = []
            for hop, (tokens, target, _) in eval_set.items():
                logits, _ = model(tokens, loops=loops)
                pred = answer_logits(logits, positions)
                accs.append((pred.argmax(-1) == target).float().mean().item())
            out[f"eval_beta_x{scale:g}"] = sum(accs) / len(accs)
    finally:
        for h, beta in zip(heads, original):
            h.cfg = replace(h.cfg, beta_scale=beta)
        model.train()
    return out


@torch.no_grad()
def attention_entropy(model, tokens, loops: Optional[int] = None):
    """Per-loop attention entropy, the order parameter of the phase diagram.

    Reported normalised by ``log(#visible keys)``: 1.0 is uniform attention
    (the disordered phase, where a head outputs an average), 0.0 is a hard
    argmax (the frozen phase, where it copies one token). The question this
    exists to answer is whether the entropy *falls across loops* -- whether a
    weight-shared head, run repeatedly on a contracting state, walks itself out
    of the useful regime. If it does, depth saturation is a head freezing, not
    only a state converging, and the two have different fixes.
    """
    from model import Attention
    heads = [m for m in model.modules() if isinstance(m, Attention)]
    for h in heads:
        h.record = True
    model.eval()
    try:
        _, aux = model(tokens, loops=loops, collect=True)
    finally:
        for h in heads:
            h.record = False
        model.train()
    per_loop = aux.get("attention_entropy") or []
    return [sum(x) / len(x) for x in per_loop if x and None not in x]


def set_active_experts(model, k: int) -> None:
    """Override every MoE layer's k for the next forward pass."""
    from model import LatentMoE
    for m in model.modules():
        if isinstance(m, LatentMoE):
            m.active_override = int(k)


def routing_state(model) -> Dict:
    """What the router settled on: mean experts per token, and the thresholds.

    Under threshold routing the mean is an *emergent* quantity, so it is the
    headline number for T4; under top-k it is a constant and is recorded only
    so the two arms can be compared at equal average sparsity.
    """
    from model import LatentMoE
    out = {"mean_k": [], "tau": [], "load": []}
    for m in model.modules():
        if isinstance(m, LatentMoE):
            out["mean_k"].append(float(m.mean_k))
            out["tau"].append(m.tau.tolist())
            out["load"].append(m.load.tolist())
    return out


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
    masks = aux.get("routing", [])
    per_loop = []
    for start in range(0, len(masks), cfg.n_core):
        counts = torch.zeros(cfg.n_routed)
        for m in masks[start:start + cfg.n_core]:
            counts += m.float().sum(0)
        per_loop.append((counts / counts.sum().clamp_min(1)).tolist())
    return per_loop


def train_hops(cfg: Config, spec: HopSpec, steps: int, batch_size: int = 64,
               lr: float = 3e-3, data_seed: int = 1234, eval_seed: int = 99,
               eval_size: int = 256, eval_every: int = 0,
               eval_loops: Sequence[int] = (),
               train_loops: Optional[Callable[[int], int]] = None,
               k_schedule: Optional[Callable[[float], int]] = None) -> Dict:
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
        if k_schedule is not None:
            # T3: the number of active experts is a *schedule*, the way
            # synaptic density is in development -- overproduce, then prune.
            set_active_experts(model, k_schedule(step / max(1, steps - 1)))
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
        "trajectories": expert_trajectories(model, eval_set, spec),
        "attention_entropy": attention_entropy(
            model, eval_set[spec.hops[0]][0][:64]),
        "beta_transfer": beta_transfer(model, eval_set, spec),
        "expert_graph": expert_graph(model, eval_set, spec),
        "routing_state": routing_state(model),
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


# ------------------------------------------------- T2/T5: the expert graph

def _loop_masks(aux, n_core: int):
    """Group per-(loop, block) routing masks into one boolean mask per loop."""
    masks = aux.get("routing", [])
    grouped = []
    for start in range(0, len(masks), n_core):
        block = masks[start:start + n_core]
        if not block:
            continue
        merged = block[0].clone()
        for m in block[1:]:
            merged |= m
        grouped.append(merged)                        # (tokens, experts) bool
    return grouped


@torch.no_grad()
def expert_trajectories(model, eval_set, spec, loops: Optional[int] = None) -> Dict:
    """T2: follow each token's expert set across loops, and ask when it settles.

    Three quantities per loop transition, all cheap:

    * **overlap** -- Jaccard between a token's expert set at loop t and t+1.
      Under the fixed-point reading this should rise toward 1.
    * **state move** -- ``||h_t - h_{t-1}|| / ||h_{t-1}||``, the actual
      contraction. If routing convergence is a read-out of state convergence
      (the linear-router argument of arXiv:2604.09780) these two curves are the
      same curve.
    * **settle step** -- the first loop after which a token's expert set never
      changes again. Reported split by whether the token's answer was correct,
      because the useful version of this is a difficulty signal: if hard
      questions settle later, the expert path is a halting rule that costs a
      set comparison rather than a softmax over the vocabulary.
    """
    model.eval()
    positions = spec.answer_positions()
    out = {}
    for hop, (tokens, target, _) in eval_set.items():
        _, aux = model(tokens, loops=loops, collect=True)
        per_loop = _loop_masks(aux, model.cfg.n_core)
        if len(per_loop) < 2:
            continue
        b, n = tokens.shape
        width = n + model.cfg.registers

        overlaps, moves = [], []
        traces = aux.get("traces", [])
        for t in range(1, len(per_loop)):
            a, c = per_loop[t - 1], per_loop[t]
            inter = (a & c).float().sum(1)
            union = (a | c).float().sum(1).clamp_min(1)
            overlaps.append((inter / union).mean().item())
            if len(traces) > t:
                delta = (traces[t] - traces[t - 1]).flatten(1).norm(dim=1)
                base = traces[t - 1].flatten(1).norm(dim=1).clamp_min(1e-6)
                moves.append((delta / base).mean().item())

        # settle step per token: last loop at which the set changed
        changed = torch.zeros(per_loop[0].shape[0])
        for t in range(1, len(per_loop)):
            differs = (per_loop[t] != per_loop[t - 1]).any(1).float()
            changed = torch.where(differs > 0, torch.full_like(changed, t + 1.0),
                                  changed)
        settle = changed.view(b, width)[:, width - n:]        # drop registers
        answer_settle = settle[:, positions]

        logits, _ = model(tokens, loops=loops)
        correct = (answer_logits(logits, positions).argmax(-1) == target)
        out[f"overlap_h{hop}"] = overlaps
        out[f"state_move_h{hop}"] = moves
        out[f"settle_h{hop}"] = answer_settle.mean().item()
        if correct.any():
            out[f"settle_correct_h{hop}"] = answer_settle[correct].mean().item()
        if (~correct).any():
            out[f"settle_wrong_h{hop}"] = answer_settle[~correct].mean().item()
        out[f"mean_k_h{hop}"] = float(
            sum(m.float().sum(1).mean().item() for m in per_loop) / len(per_loop))
    model.train()
    return out


def _modularity(adj: torch.Tensor, part: torch.Tensor) -> float:
    """Newman modularity of a two-way split -- enough to say "modular or not"."""
    m2 = adj.sum().clamp_min(1e-9)
    deg = adj.sum(1)
    same = (part.unsqueeze(0) == part.unsqueeze(1)).float()
    expect = torch.outer(deg, deg) / m2
    return float(((adj - expect) * same).sum() / m2)


@torch.no_grad()
def expert_graph(model, eval_set, spec, loops: Optional[int] = None) -> Dict:
    """T5: the expert co-activation graph, per loop.

    Nodes are experts, edge weights are how often two experts are selected for
    the same token. Three read-outs per loop: the **spectral gap** of the
    normalised Laplacian (how well connected the graph is), the **modularity**
    of its spectral two-way split (whether experts form communities), and the
    **effective number of experts** ``exp(H)``. The hypothesis this exists to
    test is that accuracy tracks structure -- a graph split into coherent
    communities -- rather than raw routing divergence, which would explain why
    E3's `embed` arm diverged four times as much and scored worse.
    """
    model.eval()
    out = {}
    hop = sorted(eval_set)[0]
    tokens = eval_set[hop][0]
    _, aux = model(tokens, loops=loops, collect=True)
    for t, mask in enumerate(_loop_masks(aux, model.cfg.n_core)):
        m = mask.float()
        adj = m.t() @ m                                  # co-activation counts
        adj.fill_diagonal_(0)
        deg = adj.sum(1)
        if float(deg.sum()) == 0:
            continue
        inv = torch.where(deg > 0, deg.clamp_min(1e-9).pow(-0.5),
                          torch.zeros_like(deg))
        lap = torch.eye(adj.shape[0]) - inv.unsqueeze(1) * adj * inv.unsqueeze(0)
        evals, evecs = torch.linalg.eigh(lap)
        share = m.sum(0) / m.sum().clamp_min(1)
        entropy = -(share * share.clamp_min(1e-12).log()).sum()
        out[f"loop{t + 1}"] = {
            "spectral_gap": float(evals[1]),
            "modularity": _modularity(adj, (evecs[:, 1] > 0).long()),
            "effective_experts": float(entropy.exp()),
            "mean_k": float(m.sum(1).mean()),
        }
    model.train()
    return out
