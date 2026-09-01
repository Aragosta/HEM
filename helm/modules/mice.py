"""Mixture of Curvature Experts (MiCE), optimized.

Each expert is a Lorentz SwiGLU MLP living on its own (optionally learnable)
curvature; a router picks ``n_activated_experts`` of them per token and a shared
expert runs on everything. The results are combined with a Lorentzian residual
so the output stays on the manifold.

Three things made the upstream dispatch slow, none of them to do with the FLOPs:

1. **A device synchronisation per expert per layer.** The loop is driven by
   ``counts = torch.bincount(...).tolist()`` (one sync) and then by
   ``torch.where(indices == i)`` inside the loop (``nonzero`` has to read the
   match count back to the host to size its output, so that is another sync per
   expert). At 16 layers x 8 experts that is ~144 full pipeline stalls per
   micro-batch, each one draining the GPU and serialising against the H2D copy
   of the next batch. This version does a single ``argsort`` + ``bincount`` and
   pays exactly **one** sync per MoE layer.

2. **A gather per expert over the whole token axis.** ``x[idx]`` builds a fresh
   fancy-index gather for every expert. Sorting the routing table once instead
   makes every expert's tokens a *contiguous slice*, so there is one gather per
   layer and the expert GEMMs read contiguous memory.

3. **Two separate GEMMs for the SwiGLU gate and up projections.** ``w1`` and
   ``w3`` consume the same input and have the same shape, so they are one
   concatenated GEMM. That halves the launch count in the part of the model that
   dominates its FLOPs -- which matters most exactly when it hurts most, i.e.
   when each expert only holds a few hundred tokens and the kernels are launch-
   bound rather than compute-bound.

Parameter *names* are preserved (``w1``/``w2``/``w3`` etc.), including for the
fused projection, via state-dict hooks -- checkpoints are interchangeable with
:mod:`helm.reference.mice` in both directions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from helm.hypercore.manifolds import Lorentz
from helm.hypercore.models.lorentz_feedforward import LorentzFeedForward
from helm.hypercore.nn.conv.conv_util_layers import LResNet
from helm.hypercore.nn.linear.lorentz_linear import LorentzLinear

world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0


class Gate(nn.Module):
    """Router producing top-k expert weights and indices.

    Unlike upstream this always returns a 3-tuple. The released ``Gate`` returns
    2 items in eval mode but ``LorentzMoE.forward`` unconditionally unpacks 3,
    so the published MiCE model raises ``ValueError`` the moment it is switched
    out of training mode -- i.e. it cannot be evaluated or generated from at all.
    """

    def __init__(self, args):
        super().__init__()
        self.dim = args.dim - 1
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.bias_update_spd = args.bias_update_speed
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, self.dim))
        # The routing bias is updated by hand in `update_bias` (DeepSeek-style
        # auxiliary-loss-free balancing), never by the optimizer.
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(tokens, dim)`` on the manifold.

        Returns:
            ``(weights, indices, scores)`` -- ``(T, topk)``, ``(T, topk)``,
            ``(T, n_experts)``. ``scores`` are the pre-bias routing
            probabilities, used by the load-balancing loss.
        """
        scores = F.linear(x[..., 1:], self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=torch.bool)
            mask = mask.scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights.type_as(x), indices, original_scores

    @torch.no_grad()
    def update_bias(self, indices: torch.Tensor):
        """Nudge the routing bias towards uniform expert utilisation.

        Args:
            indices: ``(tokens, topk)`` expert ids routed this step, **or** a
                precomputed ``(n_experts,)`` utilisation histogram.
        """
        if indices.numel() == 0:
            return
        if indices.dim() == 1 and indices.numel() == self.bias.numel():
            util = indices.to(self.bias)
        else:
            util = torch.bincount(indices.flatten(),
                                  minlength=self.bias.numel()).to(self.bias)
        total = util.sum()
        if total == 0:
            return
        util = util / total
        self.bias += self.bias_update_spd * (util.mean() - util)


class LorentzSwiGLU(nn.Module):
    """Lorentz SwiGLU MLP with the gate and up projections fused into one GEMM.

    Mathematically identical to upstream's ``w1``/``w2``/``w3`` block::

        h = silu(w1(x)) * w3(x);  out = w2([sqrt(|h|^2 + c), h])

    but ``w1`` and ``w3`` are stored as a single ``(2 * (inter_dim - 1), dim)``
    weight and split after the matmul. State-dict hooks translate to and from the
    upstream ``w1.linear.*`` / ``w3.linear.*`` layout, so checkpoints move freely
    in both directions.

    Args:
        manifold: manifold of the *output*.
        dim: input/output width (including the time coordinate).
        inter_dim: hidden width (including the time coordinate).
        expert_manifold: manifold the expert itself computes on. Defaults to
            ``manifold``; when different, inputs are rescaled onto it first.
        time_eps: ``clamp_min`` floor for the time coordinate. Upstream uses
            ``1e-6`` in the dense feed-forward and ``1e-8`` in the experts; the
            default preserves whichever the caller asks for.
    """

    def __init__(self, manifold, dim: int, inter_dim: int, expert_manifold=None,
                 time_eps: float = 1e-6):
        super().__init__()
        self.manifold = manifold
        self.c = manifold.c
        # Only register a *distinct* expert manifold. Binding `manifold` under a
        # second attribute would register its curvature twice and add state-dict
        # keys that upstream's LorentzFeedForward does not have.
        self.same_manifold = expert_manifold is None or expert_manifold is manifold
        if not self.same_manifold:
            self.expert_manifold = expert_manifold
        expert_manifold = manifold if self.same_manifold else expert_manifold
        self.dim = dim
        self.inter_dim = inter_dim
        self.time_eps = time_eps

        self.w13 = nn.Linear(dim, 2 * (inter_dim - 1), bias=True)
        # `manifold_out` only has an effect when the expert computes on a
        # different curvature than it outputs on; leaving it None otherwise
        # matches upstream's LorentzFeedForward and skips a no-op rescale.
        self.w2 = LorentzLinear(expert_manifold, inter_dim, dim - 1,
                                manifold_out=None if self.same_manifold else manifold)
        self._reset_w13()
        self._register_load_state_dict_pre_hook(self._load_upstream_layout)

    def _reset_w13(self):
        """Initialise each half exactly as two independent ``LorentzLinear`` would.

        Xavier's bound depends on ``fan_out``, so initialising the fused
        ``2 * (inter_dim - 1)`` matrix in one shot would give a different (and
        narrower) distribution than upstream's two separate draws.
        """
        out = self.inter_dim - 1
        for half in (self.w13.weight[:out], self.w13.weight[out:]):
            nn.init.xavier_uniform_(half, gain=2.0 ** 0.5)
        nn.init.constant_(self.w13.bias, 0)

    # ------------------------------------------------- checkpoint compatibility

    def _load_upstream_layout(self, state_dict, prefix, *args, **kwargs):
        """Accept upstream ``w1``/``w3`` checkpoints by concatenating them."""
        wk1, wk3 = prefix + "w1.linear.weight", prefix + "w3.linear.weight"
        bk1, bk3 = prefix + "w1.linear.bias", prefix + "w3.linear.bias"
        if wk1 not in state_dict or wk3 not in state_dict:
            return
        state_dict[prefix + "w13.weight"] = torch.cat(
            [state_dict.pop(wk1), state_dict.pop(wk3)], dim=0)
        if bk1 in state_dict and bk3 in state_dict:
            state_dict[prefix + "w13.bias"] = torch.cat(
                [state_dict.pop(bk1), state_dict.pop(bk3)], dim=0)
        # `w1`/`w3` were LorentzLinear modules, so they also carried aliases of
        # the curvature (`w1.c`, `w1.manifold.k`, ...). Those tensors are shared
        # with the manifold that is still registered elsewhere, so dropping the
        # aliases loses nothing and keeps strict=True loading working.
        for key in [k for k in state_dict
                    if k.startswith(prefix + "w1.") or k.startswith(prefix + "w3.")]:
            state_dict.pop(key)

    def upstream_state_dict(self, prefix: str = "") -> dict:
        """This module's weights in the upstream ``w1``/``w2``/``w3`` layout."""
        out = self.inter_dim - 1
        sd = {f"{prefix}w1.linear.weight": self.w13.weight[:out].detach().clone(),
              f"{prefix}w1.linear.bias": self.w13.bias[:out].detach().clone(),
              f"{prefix}w3.linear.weight": self.w13.weight[out:].detach().clone(),
              f"{prefix}w3.linear.bias": self.w13.bias[out:].detach().clone()}
        for k, v in self.w2.state_dict().items():
            sd[f"{prefix}w2.{k}"] = v.detach().clone()
        return sd

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_manifold = self.manifold if self.same_manifold else self.expert_manifold
        if not self.same_manifold:
            x = x * (expert_manifold.c / self.c).sqrt()
        gate, up = self.w13(x).chunk(2, dim=-1)
        x_space = F.silu(gate) * up
        x_time = (x_space.square().sum(dim=-1, keepdim=True)
                  + expert_manifold.c).clamp_min(self.time_eps).sqrt()
        return self.w2(torch.cat([x_time, x_space], dim=-1))


class LorentzMoE(nn.Module):
    """Mixture of Curvature Experts.

    Args:
        manifold: manifold the inputs and outputs live on.
        args: model config.
        fuse_experts: use :class:`LorentzSwiGLU` (fused gate/up GEMM) for the
            experts and the shared expert. ``False`` falls back to upstream's
            three-``LorentzLinear`` layout.
    """

    def __init__(self, manifold: Lorentz, args, fuse_experts: bool = True):
        super().__init__()
        self.dim = args.dim
        self.manifold = manifold
        if args.n_routed_experts % world_size != 0:
            raise ValueError(
                f"n_routed_experts ({args.n_routed_experts}) must be divisible by "
                f"world_size ({world_size})")
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.curvature_list = np.linspace(0.1, 2.0, self.n_routed_experts).tolist()
        self.n_shared_experts = args.n_shared_experts

        # A plain list, as upstream: each manifold is registered (and so trained
        # and checkpointed) through the expert that holds it, and putting them in
        # a ModuleList as well would duplicate every curvature in the state dict.
        self.expert_manifolds = [
            Lorentz(c=1.0, learnable=bool(args.train_curv)) for _ in range(self.n_routed_experts)]
        if fuse_experts:
            self.experts = nn.ModuleList([
                LorentzSwiGLU(self.manifold, args.dim, args.moe_inter_dim,
                              self.expert_manifolds[i], time_eps=1e-8)
                for i in range(self.n_routed_experts)])
            self.shared_experts = LorentzSwiGLU(
                self.manifold, args.dim, args.n_shared_experts * args.moe_inter_dim,
                time_eps=1e-6)
        else:
            self.experts = nn.ModuleList([
                _UnfusedExpert(self.manifold, args.dim, args.moe_inter_dim,
                               self.expert_manifolds[i])
                for i in range(self.n_routed_experts)])
            self.shared_experts = LorentzFeedForward(
                self.manifold, args.dim, args.n_shared_experts * args.moe_inter_dim)

        if self.n_activated_experts == 2:
            self.add_experts = LResNet(self.manifold, use_scale=True, scale=2.0, learn_scale=True)
        if self.n_shared_experts == 1:
            self.weighted_sum = LResNet(self.manifold, weight=1.0, use_scale=True,
                                        scale=2.0, learn_scale=False)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x_time = (x.square().sum(dim=-1, keepdim=True) + self.manifold.c).clamp_min(1e-12).sqrt()
        return torch.cat([x_time, x], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: ``(..., dim)`` on the manifold.

        Returns:
            The transformed tensor; additionally ``(indices, scores)`` while
            training, for the load-balancing loss.
        """
        shape = x.size()
        x = x.reshape(-1, self.dim)
        weights, indices, scores = self.gate(x)
        y = self.project(torch.zeros_like(x[..., 1:]))

        topk = indices.size(1)
        flat = indices.reshape(-1)
        # One sort groups every token by expert, so each expert's rows become a
        # contiguous slice: one gather for the whole layer instead of one per
        # expert, and no `nonzero` (hence no per-expert device sync).
        order = torch.argsort(flat, stable=True)
        token_of_slot = torch.div(order, topk, rounding_mode="floor")
        counts = torch.bincount(flat, minlength=self.n_routed_experts)
        xs = x.index_select(0, token_of_slot)
        ws = weights.reshape(-1).index_select(0, order).unsqueeze(-1)

        # The single unavoidable host sync of this layer: variable-sized expert
        # groups cannot be sliced without knowing their sizes on the host.
        sizes = counts.tolist()

        start = 0
        for i in range(self.experts_start_idx, self.experts_end_idx):
            n = sizes[i]
            if n == 0:
                start += n
                continue
            sl = slice(start, start + n)
            start += n
            rows = token_of_slot[sl]
            out_i = self.experts[i](xs[sl])
            if self.n_activated_experts == 2:
                # Upstream folds each expert into `y` with a Lorentzian residual,
                # which is *not* associative, so the ascending-expert order has
                # to be preserved exactly rather than summed in one shot.
                y.index_copy_(0, rows, self.weighted_sum(
                    y.index_select(0, rows), out_i, weight=ws[sl]))
            else:
                y.index_add_(0, rows, out_i * ws[sl])

        z = self.shared_experts(x)
        if self.n_shared_experts == 1:
            out = self.add_experts(z, y)
        else:
            ave = z + y
            d = ave.size(-1) - 1
            sq = ave.square()
            inner = -sq.narrow(-1, 0, 1) + sq.narrow(-1, 1, d).sum(dim=-1, keepdim=True)
            denom = inner.neg().abs().clamp_min(1e-8).sqrt()
            # NOTE(upstream bug): this branch reads `self.c`, which the released
            # LorentzMoE never assigns, so it raises AttributeError for any
            # n_shared_experts != 1. Read it off the manifold instead.
            out = self.manifold.c.sqrt() * ave / denom

        out = out.reshape(shape)
        if self.training:
            return out, indices, scores
        return out


class _UnfusedExpert(nn.Module):
    """Upstream's three-``LorentzLinear`` expert; used when ``fuse_experts=False``."""

    def __init__(self, manifold, dim: int, inter_dim: int, expert_manifold=None):
        super().__init__()
        self.c = manifold.c
        self.manifold = manifold
        self.expert_manifold = expert_manifold if expert_manifold is not None else manifold
        self.w1 = LorentzLinear(self.expert_manifold, dim, inter_dim - 1)
        self.w2 = LorentzLinear(self.expert_manifold, inter_dim, dim - 1,
                                manifold_out=self.manifold)
        self.w3 = LorentzLinear(self.expert_manifold, dim, inter_dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * (self.expert_manifold.c / self.c).sqrt()
        x_space = F.silu(self.w1(x, return_space=True)) * self.w3(x, return_space=True)
        x_time = (x_space.square().sum(dim=-1, keepdim=True)
                  + self.expert_manifold.c).clamp_min(1e-8).sqrt()
        return self.w2(torch.cat([x_time, x_space], dim=-1))
