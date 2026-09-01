#!/usr/bin/env python3
"""Verify our energy score against CALM's reference implementation.

Both are run with the target noise injected, so they draw identical samples and
any difference is the formula rather than the RNG. (Without injection they differ
only because ours matches the noise dtype to `mean` and CALM's does not, which
consumes the RNG stream differently.)

Result: bit-identical values and gradients at beta = 1.0 and 1.5. At beta = 0.5
both produce NaN gradients -- the pairwise term includes the self-distances
||x_i - x_i|| = 0, and d/dx ||x||^beta is unbounded at 0 for beta < 1. CALM's
default is 1.0, where PyTorch's subgradient for the norm at 0 is 0 and it is
safe, so nothing is broken in practice; but beta < 1 is a trap.

Usage: python CALM/experiments/verify_energy.py
"""

import torch

def calm_original(x, mean, log_std, beta, eps):
    """CALM's energy_score verbatim, with the target noise injected."""
    def distance(a, b): return torch.pow(torch.linalg.norm(a - b, ord=2, dim=-1), beta)
    n_x = x.shape[0]
    distance_matrix = distance(x.unsqueeze(1), x.unsqueeze(0))
    distance_x = distance_matrix.sum(dim=(0,1)) / (n_x * (n_x - 1))
    std = torch.exp(log_std)
    n_y = eps.shape[0]
    y = mean + eps * std
    distance_y = distance(x.reshape(n_x,1,*x.shape[1:]), y.reshape(1,n_y,*y.shape[1:])).mean(dim=(0,1))
    return distance_x - distance_y * 2

def ours(x, mean, log_std, beta, eps):
    """helm energy_score with the same injection, to isolate the formula."""
    def distance(a, b): return torch.linalg.norm(a - b, ord=2, dim=-1).pow(beta)
    n_x = x.shape[0]
    pairwise = distance(x.unsqueeze(1), x.unsqueeze(0))
    distance_x = pairwise.sum(dim=(0,1)) / (n_x * (n_x - 1))
    targets = mean + eps * log_std.exp()
    cross = distance(x.reshape(n_x,1,*x.shape[1:]), targets.reshape(1,eps.shape[0],*targets.shape[1:]))
    return distance_x - cross.mean(dim=(0,1)) * 2

torch.manual_seed(0)
n_x, T, L = 8, 40, 32
x = torch.randn(n_x, T, L, dtype=torch.float64, requires_grad=True)
mean = torch.randn(T, L, dtype=torch.float64)
log_std = torch.randn(T, L, dtype=torch.float64) * 0.2
eps = torch.randn(100, T, L, dtype=torch.float64)

for beta in (0.5, 1.0, 1.5):
    a = calm_original(x, mean, log_std, beta, eps)
    b = ours(x, mean, log_std, beta, eps)
    ga = torch.autograd.grad(a.sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(b.sum(), x, retain_graph=True)[0]
    print(f"beta={beta}: value max|diff| {(a-b).abs().max().item():.3e}   grad max|diff| {(ga-gb).abs().max().item():.3e}")
