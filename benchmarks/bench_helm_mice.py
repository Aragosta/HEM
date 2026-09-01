#!/usr/bin/env python3
"""Benchmark the optimized HELM-MiCE against the published implementation.

Times a forward pass and a full training step (forward + backward) for both, at
whatever shape you ask for, and reports wall time, throughput and peak memory.

    python benchmarks/bench_helm_mice.py                      # 120M shape
    python benchmarks/bench_helm_mice.py --preset 1b --dtype bfloat16
    python benchmarks/bench_helm_mice.py --seq-len 4096 --batch-size 2
    python benchmarks/bench_helm_mice.py --component attention # attention only

On CPU this still shows the memory and kernel-count effects, but the attention
speedup is understated: there is no FlashAttention kernel to dispatch to, so the
fused path falls back to a math implementation that still materialises scores.
Run it on a GPU for a representative number.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from helm.modules.hmla import LorentzMLA  # noqa: E402
from helm.modules.rope import precompute_freqs_cis, precompute_rope_cache  # noqa: E402
from helm.reference.helm_mice import LorentzDeepSeekV3 as RefModel  # noqa: E402
from helm.reference.hmla import LorentzMLA as RefMLA  # noqa: E402
from tests._config import bench_args, tiny_args  # noqa: E402

PRESETS = {
    "tiny": dict(dim=390, inter_dim=1560, moe_inter_dim=780, n_layers=2, n_heads=6,
                 n_routed_experts=4, kv_lora_rank=65, qk_nope_head_dim=33,
                 qk_rope_head_dim=17, v_head_dim=33, vocab_size=8192),
    "120m": {},  # bench_args defaults
    "1b": dict(dim=910, inter_dim=3640, moe_inter_dim=1820, n_layers=16, n_heads=14,
               n_routed_experts=8, kv_lora_rank=257, qk_nope_head_dim=65,
               qk_rope_head_dim=65, v_head_dim=65, vocab_size=128256),
}


def cast_module(module, dtype):
    """``module.to(dtype)``, but leave complex buffers alone.

    The upstream model stores its rotary table as a complex tensor; a plain
    ``.to(dtype)`` casts it to a real dtype and silently discards ``sin``,
    which would make the reference both wrong and unfairly fast here.
    """
    for param in module.parameters():
        if param.is_floating_point():
            param.data = param.data.to(dtype)
    for name, buf in module.named_buffers():
        if buf.is_floating_point():
            owner = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            setattr(owner, name.rsplit(".", 1)[-1], buf.to(dtype))
    return module


def count_ops(fn):
    """Number of dispatched aten ops in one call, and peak CPU memory (bytes).

    A device-independent proxy for launch overhead: on a GPU each of these is a
    kernel launch, and MoE dispatch is launch-bound far more often than it is
    compute-bound.
    """
    from torch.profiler import ProfilerActivity, profile
    fn()  # warm up lazy init out of the measurement
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    events = prof.key_averages()
    n_ops = sum(e.count for e in events if e.key.startswith("aten::"))
    peak = max((e.cpu_memory_usage for e in events), default=0)
    return n_ops, peak


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def peak_memory(device):
    """Yield a one-element list that receives peak allocated bytes."""
    out = [0]
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        yield out
        torch.cuda.synchronize()
        out[0] = torch.cuda.max_memory_allocated()
    else:
        yield out


def timeit(fn, device, warmup: int, iters: int):
    """Return (median seconds per call, peak bytes)."""
    for _ in range(warmup):
        fn()
    sync(device)
    gc.collect()
    times = []
    with peak_memory(device) as peak:
        for _ in range(iters):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], peak[0]


def fmt_mem(nbytes: int) -> str:
    return "n/a" if not nbytes else f"{nbytes / 2**20:8.0f} MiB"


def report(name: str, ref_t: float, fast_t: float, ref_m: int, fast_m: int,
           tokens: int):
    speedup = ref_t / fast_t if fast_t else float("nan")
    print(f"  {name:<22s} reference {ref_t * 1e3:9.2f} ms   "
          f"optimized {fast_t * 1e3:9.2f} ms   speedup {speedup:5.2f}x")
    print(f"  {'':<22s} {tokens / ref_t:9.0f} tok/s  ->  {tokens / fast_t:9.0f} tok/s"
          f"          peak {fmt_mem(ref_m)} -> {fmt_mem(fast_m)}")


def build_args(cli):
    base = tiny_args if cli.preset == "custom" else bench_args
    args = base(**PRESETS.get(cli.preset, {}))
    args.max_batch_size = cli.batch_size
    args.max_seq_len = cli.seq_len
    args.original_seq_len = cli.seq_len
    if cli.layers is not None:
        args.n_layers = cli.layers
    return args


def bench_attention(args, cli, device, dtype):
    print("\nHMLA (single layer)")
    manifold = Lorentz(1.0).to(device)
    ref = cast_module(RefMLA(manifold, args).to(device), dtype).eval()
    fast = LorentzMLA(manifold, args).to(device).eval()
    fast.load_state_dict(ref.state_dict())
    fast = cast_module(fast, dtype)

    b, n = cli.batch_size, cli.seq_len
    space = torch.randn(b, n, args.dim - 1, device=device, dtype=dtype) * 0.3
    x = torch.cat([(space.float().square().sum(-1, keepdim=True) + 1).sqrt().to(dtype), space], -1)
    causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)
    freqs = precompute_freqs_cis(args)[:n].to(device)

    with torch.no_grad():
        run_ref = lambda: ref(x, 0, freqs, causal)
        run_fast = lambda: fast(x, 0, freqs, mask=None, is_causal=True)
        t_ref, m_ref = timeit(run_ref, device, cli.warmup, cli.iters)
        t_fast, m_fast = timeit(run_fast, device, cli.warmup, cli.iters)
        report("forward", t_ref, t_fast, m_ref, m_fast, b * n)
        if cli.profile:
            (o_r, p_r), (o_f, p_f) = count_ops(run_ref), count_ops(run_fast)
            print(f"  {'':<22s} aten ops {o_r:6d} -> {o_f:6d}   "
                  f"peak alloc {p_r / 2**20:7.1f} -> {p_f / 2**20:7.1f} MiB")


def bench_model(args, cli, device, dtype):
    manifolds = [Lorentz(1.0).to(device) for _ in range(3)]
    ref = cast_module(RefModel(args, *manifolds).to(device), dtype)
    fast = HelmMiCE(args, *manifolds, grad_checkpoint=cli.grad_checkpoint).to(device)
    fast.load_state_dict(ref.state_dict(), strict=False)
    fast = cast_module(fast, dtype)
    n_params = sum(p.numel() for p in fast.parameters())
    print(f"\nHELM-MiCE  ({n_params / 1e6:.1f}M params, {args.n_layers} layers, "
          f"dim={args.dim}, {args.n_routed_experts} experts)")

    b, n = cli.batch_size, cli.seq_len
    tokens = torch.randint(0, args.vocab_size, (b, n), device=device)

    ref.train()
    fast.train()

    labels = tokens.roll(-1, 1).clone()
    labels[:, -1] = -100
    labels[labels % 7 == 0] = -100          # what sequence packing produces

    def step(model, fused_head=False):
        """One training step: forward, cross-entropy, backward."""
        def run():
            model.zero_grad(set_to_none=True)
            if fused_head:
                loss = model(tokens, labels=labels,
                             ce_chunk_size=cli.ce_chunk_size)[0]
            else:
                logits = model(tokens)[0]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                    ignore_index=-100)
            loss.backward()
        return run

    def fwd(model, **kw):
        def run():
            with torch.no_grad():
                model(tokens, **kw)
        return run

    t_ref, m_ref = timeit(fwd(ref), device, cli.warmup, cli.iters)
    t_fast, m_fast = timeit(fwd(fast), device, cli.warmup, cli.iters)
    report("forward (train mode)", t_ref, t_fast, m_ref, m_fast, b * n)

    t_ref, m_ref = timeit(step(ref), device, cli.warmup, cli.iters)
    t_fast, m_fast = timeit(step(fast, fused_head=True), device, cli.warmup, cli.iters)
    report("training step", t_ref, t_fast, m_ref, m_fast, b * n)

    if cli.profile:
        (o_r, p_r), (o_f, p_f) = count_ops(step(ref)), count_ops(step(fast))
        print(f"  {'':<22s} aten ops {o_r:6d} -> {o_f:6d}   "
              f"peak alloc {p_r / 2**20:7.1f} -> {p_f / 2**20:7.1f} MiB")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="120m", choices=sorted(PRESETS) + ["custom"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--layers", type=int, default=None, help="override n_layers")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--component", default="all", choices=["all", "attention", "model"])
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--ce-chunk-size", type=int, default=512)
    p.add_argument("--profile", action="store_true",
                   help="also report dispatched op counts and peak allocation")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    cli = p.parse_args()

    device = torch.device(cli.device)
    dtype = getattr(torch, cli.dtype)
    torch.manual_seed(0)

    args = build_args(cli)
    print(f"device={device}  dtype={cli.dtype}  batch={cli.batch_size}  "
          f"seq_len={cli.seq_len}  preset={cli.preset}")
    if device.type == "cpu":
        print("NOTE: on CPU there is no FlashAttention kernel, so the attention "
              "speedup below is a lower bound.")

    if cli.component in ("all", "attention"):
        bench_attention(args, cli, device, dtype)
    if cli.component in ("all", "model"):
        bench_model(args, cli, device, dtype)


if __name__ == "__main__":
    main()
