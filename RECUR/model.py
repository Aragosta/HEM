"""The baseline backbone and every recurrence variant, in one place.

The baseline is a scaled-down **Kimi K3** block: a Mixture-of-Experts
transformer with **multi-head attention** and **Attention Residuals**
(`BASELINE.md` records what was ported faithfully, what was substituted, and
what could not be established from public sources). Everything this study
tests is a modification of that block, and every modification is switchable
from :class:`Config` so that two arms can be built from the same code path and
differ in exactly one field.

The rule the whole suite depends on: **an arm is a config, not a class.** If a
variant needed its own module the comparison would silently pick up whatever
else that module did differently. So the loop, the register bank, the halting
gate and the routing conditioner are all flags on one model, and
:func:`Config.differs_from` prints the fields two arms actually differ in --
which is what gets recorded in every result file.

Layer indexing follows the AttnRes report: a *layer* is one sublayer, so a
transformer block is two of them (attention, then MLP). ``n_layers`` counts
blocks; ``2 * n_layers`` counts AttnRes sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------- config

@dataclass
class Config:
    """One arm. Two arms differ by the fields, and only the fields, listed."""

    vocab_size: int = 256
    dim: int = 192
    n_heads: int = 6
    max_seq_len: int = 512

    # --- Kimi K3 baseline block -------------------------------------------
    # MoE: routed experts in a down-projected latent space, plus shared
    # experts that every token sees (K3: 16 of 896 routed + 2 shared).
    moe: bool = True
    n_routed: int = 16
    n_active: int = 2
    n_shared: int = 1
    latent_ratio: float = 0.5          # d_latent / d_model (K3: 3584/7168)
    expert_ratio: float = 0.86         # d_ff / d_latent   (K3: 3072/3584)
    dense_ratio: float = 2.67          # d_ff / d_model when moe=False
    router_bias_lr: float = 1e-3       # aux-loss-free load balancing
    attn_res: str = "full"             # "none" | "full"
    sandwich_norm: bool = True

    # --- recurrence --------------------------------------------------------
    # prelude -> [core] x R -> coda. R = 1 with n_prelude = n_coda = 0 is the
    # plain non-recurrent transformer.
    n_prelude: int = 1
    n_core: int = 2
    n_coda: int = 1
    loops: int = 1
    inject_input: bool = True          # Huginn: re-inject the prelude output
    random_state_init: bool = True     # Huginn: random latent, not zeros
    backprop_loops: int = 0            # 0 = all; k = last k loops only

    # --- what the loop may write to ---------------------------------------
    registers: int = 0                 # extra writable positions in the state
    register_persist: bool = True      # False = same width, wiped each loop
    loop_memory: str = "none"          # "none" | "attn_res" across loops
    loop_attn_res: str = "shared"      # "shared" | "per_step" pseudo-queries

    # --- MoE routing under recurrence -------------------------------------
    step_routing: str = "none"         # "none" | "embed" | "bias"

    # --- halting -----------------------------------------------------------
    halting: str = "none"              # "none" | "ouro" | "pondernet"
    halt_prior: float = 0.5            # geometric lambda (pondernet)
    halt_beta: float = 0.01            # prior regulariser weight
    max_train_loops: int = 4           # T for the step-indexed gate

    seed: int = 0

    # ---------------------------------------------------------------- derived
    @property
    def d_latent(self) -> int:
        return max(self.n_heads, int(round(self.dim * self.latent_ratio)))

    @property
    def d_expert(self) -> int:
        return max(8, int(round(self.d_latent * self.expert_ratio)))

    @property
    def d_dense(self) -> int:
        return max(8, int(round(self.dim * self.dense_ratio)))

    @property
    def n_blocks(self) -> int:
        """Blocks actually executed on a forward pass, at ``loops``."""
        return self.n_prelude + self.loops * self.n_core + self.n_coda

    def differs_from(self, other: "Config") -> Dict[str, Tuple[object, object]]:
        a, b = asdict(self), asdict(other)
        return {k: (b[k], a[k]) for k in a if a[k] != b[k]}

    def but(self, **kw) -> "Config":
        return replace(self, **kw)


# ------------------------------------------------------------------- pieces

def rotary_table(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    paired = torch.view_as_complex(x.float().reshape(*shape[:-1], -1, 2))
    rotated = paired * table[: shape[-2]].view(1, 1, shape[-2], -1)
    return torch.view_as_real(rotated).reshape(*shape).type_as(x)


class Attention(nn.Module):
    """Plain causal MHA with RoPE.

    K3 ships 69 Kimi Delta Attention layers to 24 gated MLA layers; this study
    was asked for the MHA configuration and MHA is what the sequence lengths
    here (<=256) can justify -- a linear-attention layer is a statement about
    long context and would be untestable at this scale. See ``BASELINE.md``.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, rotary, attn_mask=None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, n, self.heads, d // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        if rotary is not None:
            q, k = apply_rotary(q, rotary), apply_rotary(k, rotary)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class LatentMoE(nn.Module):
    """K3's LatentMoE: route in a down-projected space, not at model width.

    ``down`` takes the token to ``d_latent`` (K3: half of ``d_model``), every
    expert is a SwiGLU inside that space, and ``up`` returns to model width.
    Shared experts run on every token; routed experts are the top ``n_active``
    of ``n_routed`` under a sigmoid router.

    Load balancing is **aux-loss-free**: a per-expert bias enters the routing
    scores but not the combining weights, and is nudged after every step
    toward equal load (DeepSeek-V3 / K2). "Stable LatentMoE" is named but not
    described in the K3 material that was reachable, so this is the standard
    mechanism of that lineage, flagged as such rather than guessed at.

    ``step_routing`` is this study's modification, not K3's: under a weight-tied
    loop, every iteration otherwise routes with the same function. "embed" adds
    a learned per-loop vector to the router input; "bias" adds a learned
    per-loop bias to the logits (weaker: it can reorder experts globally but
    cannot make the ordering depend on the token differently per loop).
    """

    def __init__(self, cfg: Config, max_steps: int):
        super().__init__()
        self.cfg = cfg
        dl, de = cfg.d_latent, cfg.d_expert
        self.down = nn.Linear(cfg.dim, dl, bias=False)
        self.up = nn.Linear(dl, cfg.dim, bias=False)
        self.n_routed, self.n_active = cfg.n_routed, cfg.n_active
        self.gate_w = nn.Parameter(torch.empty(cfg.n_routed, dl))
        nn.init.normal_(self.gate_w, std=dl ** -0.5)
        self.register_buffer("gate_bias", torch.zeros(cfg.n_routed))
        self.register_buffer("load", torch.zeros(cfg.n_routed))
        # experts as batched parameters: (E, dl, de) so top-k is a gather
        self.w_gate = nn.Parameter(torch.empty(cfg.n_routed, dl, de))
        self.w_up = nn.Parameter(torch.empty(cfg.n_routed, dl, de))
        self.w_down = nn.Parameter(torch.empty(cfg.n_routed, de, dl))
        for t in (self.w_gate, self.w_up):
            nn.init.normal_(t, std=dl ** -0.5)
        nn.init.normal_(self.w_down, std=de ** -0.5)
        if cfg.n_shared:
            self.s_gate = nn.Linear(dl, de * cfg.n_shared, bias=False)
            self.s_up = nn.Linear(dl, de * cfg.n_shared, bias=False)
            self.s_down = nn.Linear(de * cfg.n_shared, dl, bias=False)
        self.step_embed = None
        self.step_bias = None
        if cfg.step_routing == "embed":
            self.step_embed = nn.Parameter(torch.zeros(max_steps, dl))
        elif cfg.step_routing == "bias":
            self.step_bias = nn.Parameter(torch.zeros(max_steps, cfg.n_routed))

    def forward(self, x: torch.Tensor, step: int = 0):
        cfg = self.cfg
        z = self.down(x)                                   # (B, N, dl)
        flat = z.reshape(-1, z.shape[-1])
        router_in = flat
        if self.step_embed is not None:
            router_in = flat + self.step_embed[min(step, self.step_embed.shape[0] - 1)]
        logits = router_in @ self.gate_w.t()
        if self.step_bias is not None:
            logits = logits + self.step_bias[min(step, self.step_bias.shape[0] - 1)]
        scores = torch.sigmoid(logits)
        idx = torch.topk(scores + self.gate_bias, self.n_active, dim=-1).indices
        weight = torch.gather(scores, 1, idx)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-9)

        # Dispatch: one matmul per expert over the tokens routed to it. The
        # alternative -- gathering (T, k, dl, de) expert weights and batching --
        # is a page of tensor algebra and 100x the memory traffic; measured at
        # these shapes it was 20x slower.
        out = torch.zeros_like(flat)
        flat_idx = idx.reshape(-1)
        flat_w = weight.reshape(-1)
        token_of = torch.arange(idx.shape[0], device=x.device
                                ).repeat_interleave(self.n_active)
        order = torch.argsort(flat_idx)
        counts = torch.bincount(flat_idx, minlength=self.n_routed).tolist()
        start = 0
        for e, n_e in enumerate(counts):
            if n_e == 0:
                continue
            sel = order[start:start + n_e]
            start += n_e
            rows = token_of[sel]
            z_e = flat[rows]
            h_e = F.silu(z_e @ self.w_gate[e]) * (z_e @ self.w_up[e])
            out.index_add_(0, rows, (h_e @ self.w_down[e]) * flat_w[sel].unsqueeze(-1))

        if cfg.n_shared:
            out = out + self.s_down(F.silu(self.s_gate(flat)) * self.s_up(flat))

        with torch.no_grad():
            counts = torch.zeros_like(self.load)
            counts.scatter_add_(0, idx.reshape(-1),
                                torch.ones(idx.numel(), device=x.device))
            self.load.mul_(0.9).add_(0.1 * counts / max(1, idx.shape[0]))
        return self.up(out.view_as(z)), idx, scores

    @torch.no_grad()
    def rebalance(self):
        """Aux-loss-free step: push the bias toward equal expert load."""
        target = self.load.mean()
        self.gate_bias.add_(self.cfg.router_bias_lr *
                            torch.sign(target - self.load))


class DenseFFN(nn.Module):
    def __init__(self, cfg: Config, max_steps: int):
        super().__init__()
        d, h = cfg.dim, cfg.d_dense
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x, step: int = 0):
        return self.down(F.silu(self.gate(x)) * self.up(x)), None, None


class Block(nn.Module):
    """One transformer block = two AttnRes sources (attention, then MLP)."""

    def __init__(self, cfg: Config, max_steps: int):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = nn.RMSNorm(cfg.dim, eps=1e-5)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.dim, eps=1e-5)
        self.mlp = LatentMoE(cfg, max_steps) if cfg.moe else DenseFFN(cfg, max_steps)
        # Ouro reports sandwich normalisation as load-bearing for recurrent
        # stability: RMSNorm on the way out as well as in.
        self.post_attn = nn.RMSNorm(cfg.dim, eps=1e-5) if cfg.sandwich_norm else None
        self.post_mlp = nn.RMSNorm(cfg.dim, eps=1e-5) if cfg.sandwich_norm else None

    def forward(self, stream, rotary, step: int, attn_mask=None):
        """``stream`` is the residual mechanism; it decides what each sublayer
        reads and records what it wrote."""
        h = stream.read(self.attn_norm, kind="attn", step=step)
        a = self.attn(h, rotary, attn_mask)
        if self.post_attn is not None:
            a = self.post_attn(a)
        stream.write(a)

        h = stream.read(self.mlp_norm, kind="mlp", step=step)
        m, idx, scores = self.mlp(h, step=step)
        if self.post_mlp is not None:
            m = self.post_mlp(m)
        stream.write(m)
        return idx, scores


# --------------------------------------------------------- residual mechanism

class Stream:
    """Standard residuals, as the control for AttnRes.

    Holds the running sum; ``read`` returns it normalised, ``write`` adds.
    """

    def __init__(self, h1: torch.Tensor, model: "Recurrent"):
        self.h = h1
        self.model = model
        self.index = 0

    def read(self, norm, kind: str, step: int):
        return norm(self.h)

    def write(self, v: torch.Tensor):
        self.h = self.h + v
        self.index += 1

    def final(self):
        return self.h


class AttnResStream(Stream):
    """Full Attention Residuals (arXiv:2603.15031, Kimi Team).

        h_l = sum_i alpha_{i->l} v_i,
        alpha_{i->l} = softmax_i( w_l . RMSNorm(v_i) ),
        v_0 = h_1 (embedding),  v_i = f_i(h_i)  for i >= 1

    with ``w_l`` a learned per-sublayer pseudo-query, **zero-initialised** so
    that training starts from a uniform average over sources (the report says
    this is required; a nonzero init made training volatile).

    Under recurrence the source list keeps growing across loops, which is the
    reason this baseline is interesting here and not merely inherited: AttnRes
    gives a looped model read access to every earlier loop's outputs, so it is
    already a partial answer to the "nowhere to write" objection (`README.md`
    section 3). ``loop_attn_res`` controls whether the pseudo-query is shared
    across loops (weight tying taken literally) or learned per loop step.
    """

    def __init__(self, h1: torch.Tensor, model: "Recurrent"):
        super().__init__(h1, model)
        self.values: List[torch.Tensor] = [h1]
        self.keys: List[torch.Tensor] = [model.src_norm(h1)]
        self.last_weights: Optional[torch.Tensor] = None

    def read(self, norm, kind: str, step: int):
        q = self.model.pseudo_query(self.index, kind, step)
        keys = torch.stack(self.keys, 0)                   # (S, B, N, D)
        logits = torch.einsum('d,sbnd->sbn', q, keys)
        alpha = logits.softmax(0)
        self.last_weights = alpha.detach()
        h = torch.einsum('sbn,sbnd->bnd', alpha, torch.stack(self.values, 0))
        self.h = h
        return norm(h)

    def write(self, v: torch.Tensor):
        self.values.append(v)
        self.keys.append(self.model.src_norm(v))
        self.index += 1

    def final(self):
        # The output layer aggregates all sources, as in the report's final
        # aggregation; reuse the pseudo-query slot reserved for the coda.
        q = self.model.pseudo_query(self.index, "final", 0)
        keys = torch.stack(self.keys, 0)
        alpha = torch.einsum('d,sbnd->sbn', q, keys).softmax(0)
        return torch.einsum('sbn,sbnd->bnd', alpha, torch.stack(self.values, 0))


# ------------------------------------------------------------------- the model

class Recurrent(nn.Module):
    """prelude -> (core x R) -> coda, with every switch in :class:`Config`.

    With ``loops=1`` and no halting this is exactly the K3-style baseline; the
    recurrence code path is still the one that runs, so the R=1 arm is not a
    different program from the R=8 arm.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        max_steps = max(cfg.loops, cfg.max_train_loops, 1)
        mk = lambda n: nn.ModuleList([Block(cfg, max_steps) for _ in range(n)])
        self.prelude, self.core, self.coda = mk(cfg.n_prelude), mk(cfg.n_core), mk(cfg.n_coda)
        self.final_norm = nn.RMSNorm(cfg.dim, eps=1e-5)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.src_norm = nn.RMSNorm(cfg.dim, eps=1e-5)

        # AttnRes pseudo-queries. One per (sublayer slot); under a loop the
        # core's slots repeat, and `loop_attn_res` says whether they repeat
        # their query too.
        n_slots = 2 * (cfg.n_prelude + cfg.n_core + cfg.n_coda) + 1
        steps = max_steps if cfg.loop_attn_res == "per_step" else 1
        self.queries = nn.Parameter(torch.zeros(steps, n_slots, cfg.dim))

        if cfg.inject_input:
            self.adapter = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)
        if cfg.registers:
            self.register_init = nn.Parameter(torch.zeros(cfg.registers, cfg.dim))
            nn.init.normal_(self.register_init, std=0.02)
        if cfg.halting == "ouro":
            self.exit_gate = nn.Linear(cfg.dim, max_steps, bias=True)
        elif cfg.halting == "pondernet":
            self.exit_gate = nn.Linear(cfg.dim, 1, bias=True)
        if cfg.halting != "none":
            nn.init.zeros_(self.exit_gate.weight)
            nn.init.zeros_(self.exit_gate.bias)

        head_dim = cfg.dim // cfg.n_heads
        self.register_buffer("rotary",
                             rotary_table(head_dim, cfg.max_seq_len + cfg.registers),
                             persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        """Fan-in scaled, not the GPT-2 constant.

        An earlier version used ``std=0.02`` everywhere, which is the
        convention for ``d_model`` around 768. At ``dim=64`` it is six times
        smaller than fan-in scaling, and the effect was not subtle: attention
        logits started near zero, attention started near uniform, and the model
        sat at chance on a task a textbook transformer with default
        initialisation solved completely (`probe_reference.py`, and the ladder
        in `RESULTS.md`). Initialisation was the whole gap.
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=m.in_features ** -0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=1.0)

    # --- AttnRes plumbing --------------------------------------------------
    def pseudo_query(self, index: int, kind: str, step: int):
        s = min(step, self.queries.shape[0] - 1) if self.queries.shape[0] > 1 else 0
        # Under a persistent cross-loop stream the source index runs past the
        # slot count; wrapping is weight tying taken literally -- the same
        # sublayer position reuses its pseudo-query on every loop. Per-step
        # queries are the untied alternative, selected by `loop_attn_res`.
        return self.queries[s, index % self.queries.shape[1]]

    def _stream(self, h1):
        if self.cfg.attn_res == "full":
            return AttnResStream(h1, self)
        return Stream(h1, self)

    # --- forward -----------------------------------------------------------
    def forward(self, tokens: torch.Tensor, loops: Optional[int] = None,
                collect: bool = False):
        """Returns ``(logits, aux)``.

        ``aux['step_logits']`` holds one logit tensor per loop when a halting
        gate is present or ``collect`` is set -- every loop is a legitimate
        output point, which is what makes early exit exact rather than
        approximate.
        """
        cfg = self.cfg
        R = cfg.loops if loops is None else loops
        b, n = tokens.shape
        x = self.embed(tokens)
        if cfg.registers:
            regs = self.register_init.unsqueeze(0).expand(b, -1, -1)
            x = torch.cat([regs, x], dim=1)
            n_total = n + cfg.registers
        else:
            n_total = n
        rotary = self.rotary[:n_total]

        # registers sit before the sequence and are visible to every token,
        # so the mask is causal on the text and open on the register block.
        attn_mask = None
        if cfg.registers:
            i = torch.arange(n_total, device=tokens.device)
            attn_mask = (i[None, :] <= i[:, None])
            attn_mask[:, :cfg.registers] = True
            attn_mask = attn_mask.view(1, 1, n_total, n_total)

        stream = self._stream(x)
        for blk in self.prelude:
            blk(stream, rotary, step=0, attn_mask=attn_mask)
        embedded = stream.final() if isinstance(stream, AttnResStream) else stream.h

        state = embedded
        if cfg.random_state_init:
            g = torch.Generator(device=state.device).manual_seed(cfg.seed + 1)
            state = torch.randn(state.shape, generator=g, device=state.device,
                                dtype=state.dtype) * 0.02

        step_logits, traces, halt = [], [], []
        cut = R - cfg.backprop_loops if cfg.backprop_loops else 0
        routing: List[torch.Tensor] = []
        persistent = cfg.loop_memory == "attn_res" and cfg.attn_res == "full"
        carried = self._stream(state) if persistent else None
        for r in range(R):
            if r < cut and not persistent:
                # Truncated backprop cuts the state; a persistent AttnRes
                # stream keeps every earlier source alive, so detaching the
                # state there would not actually cut the graph. The two
                # features are therefore not combined, rather than combined
                # and quietly ineffective.
                state = state.detach()
            if cfg.registers and not cfg.register_persist:
                state = torch.cat([
                    self.register_init.unsqueeze(0).expand(b, -1, -1),
                    state[:, cfg.registers:]], dim=1)
            if persistent:
                if cfg.inject_input:
                    # Re-injection as a written source: the loop's reads are a
                    # softmax over sources, so an additive injection would be
                    # discarded. The input is in any case always source 0.
                    carried.write(self.adapter(
                        torch.cat([embedded, carried.final()], dim=-1)))
                loop_stream = carried
            else:
                if cfg.inject_input:
                    state = self.adapter(torch.cat([embedded, state], dim=-1))
                loop_stream = self._stream(state)
            for blk in self.core:
                idx, _ = blk(loop_stream, rotary, step=r, attn_mask=attn_mask)
                if idx is not None and collect:
                    routing.append(idx.detach())
            state = loop_stream.final()
            if collect:
                traces.append(state.detach())
            if cfg.halting != "none" or collect:
                step_logits.append(self._decode(state, rotary, attn_mask, n))
            if cfg.halting == "ouro":
                halt.append(torch.sigmoid(self.exit_gate(state.mean(1))[:, min(r, self.exit_gate.out_features - 1)]))
            elif cfg.halting == "pondernet":
                halt.append(torch.sigmoid(self.exit_gate(state.mean(1)).squeeze(-1)))

        logits = step_logits[-1] if step_logits else self._decode(state, rotary, attn_mask, n)
        aux = {"step_logits": step_logits, "halt": halt}
        if collect:
            aux["traces"] = traces
            aux["routing"] = routing
        return logits, aux

    def _decode(self, state, rotary, attn_mask, n: int):
        stream = self._stream(state)
        for blk in self.coda:
            blk(stream, rotary, step=0, attn_mask=attn_mask)
        h = stream.final()
        h = self.final_norm(h)
        if self.cfg.registers:
            h = h[:, self.cfg.registers:]
        return self.head(h)

    # --- housekeeping ------------------------------------------------------
    def rebalance(self):
        for m in self.modules():
            if isinstance(m, LatentMoE):
                m.rebalance()

    def expert_load(self) -> List[torch.Tensor]:
        return [m.load.clone() for m in self.modules() if isinstance(m, LatentMoE)]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_active_params(self) -> int:
        """Parameters touched per token at the configured loop count.

        A looped model reuses its core, so active-per-token counts the core
        once even though it runs R times -- FLOPs are the quantity that scales
        with R, and :func:`flops_per_token` reports that separately. Reporting
        one without the other is how a looped model gets to look free.
        """
        total = 0
        for name, p in self.named_parameters():
            if name.split(".")[-1] in ("w_gate", "w_up", "w_down"):
                continue
            total += p.numel()
        cfg = self.cfg
        moe_per_block = 0
        for m in self.modules():
            if isinstance(m, LatentMoE):
                moe_per_block = (cfg.n_active *
                                 (m.w_gate[0].numel() + m.w_up[0].numel() +
                                  m.w_down[0].numel()))
                break
        return total + moe_per_block * (cfg.n_prelude + cfg.n_core + cfg.n_coda)

    def flops_per_token(self, seq_len: int) -> float:
        """A transparent 2*MACs count for the executed blocks.

        Attention score/value FLOPs use ``seq_len`` explicitly, so R and
        sequence length trade off honestly against each other.
        """
        cfg = self.cfg
        d = cfg.dim
        per_block = 2 * (4 * d * d)                      # qkv + out projection
        per_block += 2 * (2 * seq_len * d)               # scores + weighted sum
        if cfg.moe:
            per_block += 2 * (2 * d * cfg.d_latent)      # down + up
            ffn = cfg.n_active + cfg.n_shared
            per_block += 2 * (3 * ffn * cfg.d_latent * cfg.d_expert)
        else:
            per_block += 2 * (3 * d * cfg.d_dense)
        blocks = cfg.n_prelude + cfg.loops * cfg.n_core + cfg.n_coda
        total = blocks * per_block
        total += 2 * d * cfg.vocab_size                  # head
        if cfg.inject_input:
            total += cfg.loops * 2 * 2 * d * d
        return float(total)
