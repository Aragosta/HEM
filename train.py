"""Train HELM-D or HELM-MiCE.

Same CLI and same training maths as the released script, with the host
synchronisations and the load-balancing bugs taken out of the inner loop. See
``docs/OPTIMIZATIONS.md``.
"""

import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from config.args import parser
from helm.hypercore.manifolds import Lorentz
from helm.modules.helm_d import LTransformerDecoder
from helm.modules.helm_mice import HelmMiCE
from helm.modules.mice import LorentzMoE
from helm.utils.train_util import (prepare_accelerator, prepare_data,
                                   save_checkpoint_both, save_checkpoint_euc)


def resolve_rope_impl(args):
    """Pick the rotary implementation that suits the execution mode.

    The complex path has the faster eager kernel, but TorchInductor cannot lower
    complex operators, so under ``torch.compile`` it falls back and drags the
    surrounding fusion down with it. Measured on one block (CPU, seq 512):

        eager    complex 37.5 ms | real 46.8 ms
        compiled complex 34.8 ms | real 30.6 ms

    So: complex when running eagerly, real when compiling. Pass ``--rope_impl``
    explicitly to override.
    """
    if args.rope_impl != "auto":
        return args.rope_impl
    return "real" if args.compile else "complex"


def sequence_balance_loss(scores: torch.Tensor, indices: torch.Tensor,
                          alpha: float) -> torch.Tensor:
    """DeepSeek-style auxiliary load-balancing loss for one MoE layer.

        L = alpha * E * sum_i f_i * P_i

    where ``f_i`` is the fraction of routing slots that went to expert ``i`` and
    ``P_i`` is its mean routing probability.

    The released implementation of this function does not compute that, and
    cannot: it builds a histogram and then immediately overwrites it with
    ``freq = indices * (E / (k * N))`` -- the *expert ids*, not their counts,
    shaped ``(N, k)`` rather than ``(E,)``. The result only broadcasts against
    ``P`` when ``k`` happens to equal ``E``, and otherwise raises. It also
    hard-codes ``k = 2`` next to the line that would have read it off ``indices``.
    Callers made it worse by round-tripping the indices through the host
    (``torch.tensor(idx, device='cpu')``), forcing a device sync per layer per
    micro-batch.

    Args:
        scores: ``(tokens, n_experts)`` routing probabilities.
        indices: ``(tokens, topk)`` selected expert ids.
        alpha: loss weight.

    Returns:
        Scalar loss on the same device as ``scores``.
    """
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    n_tokens, n_experts = scores.shape
    topk = indices.size(1)
    counts = torch.bincount(indices.reshape(-1), minlength=n_experts).to(scores.dtype)
    freq = counts * (n_experts / (topk * n_tokens))
    probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return alpha * (freq * probs.mean(dim=0)).sum()


def build_model(args):
    manifold_in, manifold_hidden, manifold_out = Lorentz(1.0), Lorentz(1.0), Lorentz(1.0)
    if args.model_name == "HELM_MiCE":
        return HelmMiCE(
            args, manifold_in, manifold_hidden, manifold_out,
            attn_impl=args.attn_impl,
            rope_impl=resolve_rope_impl(args),
            fuse_experts=args.fuse_experts,
            fuse_residual=args.fuse_residual,
            grad_checkpoint=args.grad_checkpoint,
        )
    if args.model_name == "HELM_D":
        return LTransformerDecoder(manifold_in, manifold_hidden, manifold_out,
                                   args.arch, args.vocab_size, args.max_seq_len)
    raise NotImplementedError(f"unknown model_name {args.model_name!r}")


def train(args, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    checkpoint_dir = args.CHECKPOINT_DIR

    print("Initializing training...")
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=args.find_unused_parameters,
        # The only buffers are the rotary table and the causal mask, both of
        # which are deterministic functions of the config and identical on every
        # rank. Broadcasting them every step is pure wire time.
        broadcast_buffers=False,
    )
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed, device_specific=True)

    print("Loading model and optimizer...")
    decoder = build_model(args)
    is_mice = args.model_name == "HELM_MiCE"
    print(f"Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # HELM-D path only

    train_dataloader = prepare_data(tokenizer, args)
    print("Dataset loaded and DataLoader prepared.")

    print("Preparing for accelerator...")
    if not args.project_emb:
        train_dataloader, decoder, scheduler_euc, scheduler_hyp, optimizer = \
            prepare_accelerator(accelerator, train_dataloader, decoder, args)
    else:
        train_dataloader, decoder, scheduler_euc, optimizer = \
            prepare_accelerator(accelerator, train_dataloader, decoder, args)
        scheduler_hyp = None

    if args.compile:
        # The MoE dispatch is data dependent (variable-sized expert groups), so
        # dynamo will break the graph there. Everything around it -- attention,
        # norms, residuals, the dense layers -- still compiles.
        decoder = torch.compile(decoder)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir) if accelerator.is_main_process else None

    decoder.train()
    global_step = 0
    total_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    progress_bar = tqdm(range(total_steps), unit="update")

    # Loss and routing statistics are accumulated *on device* and read back once
    # per optimizer step. The released loop called `.item()` on every micro-batch
    # and moved a per-layer histogram to the host with `.cpu()`, so each
    # micro-batch ended in a full pipeline stall.
    running_loss = torch.zeros((), device=accelerator.device)
    expert_usage = None

    for batch in train_dataloader:
        seq_ids = batch["sequence_id"]
        block_mask = ~(seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2))
        with accelerator.accumulate(decoder):
            if is_mice:
                # `labels=` runs the fused head: the (batch, seq, vocab) logit
                # tensor -- 3.9 GiB at the released shape -- is never built, and
                # ignored positions skip the projection entirely.
                loss, indices_list, scores_list = decoder(
                    batch["input_ids"], attn_mask=block_mask,
                    labels=batch["labels"], ce_chunk_size=args.ce_chunk_size)
            else:
                logits = decoder(batch["input_ids"], attn_mask=block_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)),
                               batch["labels"].view(-1))

            if is_mice and indices_list:
                balance = torch.stack([
                    sequence_balance_loss(scr, idx, args.seq_bal_alpha)
                    for idx, scr in zip(indices_list, scores_list)]).sum()
                loss = loss + balance
                if args.balance_update:
                    if expert_usage is None:
                        expert_usage = torch.zeros(len(indices_list), args.n_routed_experts,
                                                   device=accelerator.device)
                    for layer_id, idx in enumerate(indices_list):
                        expert_usage[layer_id] += torch.bincount(
                            idx.reshape(-1), minlength=args.n_routed_experts).to(expert_usage)

            accelerator.backward(loss)
            running_loss += loss.detach()

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                scheduler_euc.step()
                if scheduler_hyp is not None:
                    scheduler_hyp.step()
                optimizer.zero_grad()

                if is_mice and args.balance_update and expert_usage is not None:
                    _update_routing_bias(accelerator, decoder, expert_usage)
                    expert_usage = None

                # The one host sync per optimizer step, instead of one per
                # micro-batch (256 of them at the released defaults).
                mean_loss = (accelerator.gather(running_loss).mean().item()
                             / accelerator.gradient_accumulation_steps)
                running_loss = torch.zeros((), device=accelerator.device)

                if writer is not None:
                    writer.add_scalar("train/loss", mean_loss, global_step)
                    writer.add_scalar("train/lr_euc", scheduler_euc.get_last_lr()[0], global_step)
                    if scheduler_hyp is not None:
                        writer.add_scalar("train/lr_hyp", scheduler_hyp.get_last_lr()[0],
                                          global_step)

                progress_bar.set_postfix({"Batch Loss": f"{mean_loss:.4f}"})
                global_step += 1
                if global_step % 100 == 0:
                    _save(accelerator, decoder, optimizer, scheduler_euc, scheduler_hyp,
                          args, checkpoint_dir, global_step)
                progress_bar.update(1)

    _save(accelerator, decoder, optimizer, scheduler_euc, scheduler_hyp, args,
          checkpoint_dir, global_step)


@torch.no_grad()
def _update_routing_bias(accelerator, decoder, expert_usage):
    """Auxiliary-loss-free expert balancing (DeepSeek-V3 section 2.1.2).

    Nudges each router's bias towards uniform utilisation, using the statistics
    gathered over the whole optimizer step.

    The released script implements this and then comments the entire block out,
    so ``Gate.update_bias`` is dead code and the bias stays at zero for the whole
    run -- the auxiliary-loss-free half of the balancing strategy never runs.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(expert_usage, op=dist.ReduceOp.SUM)
    model = accelerator.unwrap_model(decoder)
    layer_id = 0
    for layer in model.layers:
        if not isinstance(layer.ffn, LorentzMoE):
            continue
        layer.ffn.gate.update_bias(expert_usage[layer_id])
        layer_id += 1


def _save(accelerator, decoder, optimizer, scheduler_euc, scheduler_hyp, args,
          checkpoint_dir, global_step):
    if scheduler_hyp is not None:
        save_checkpoint_both(accelerator, decoder, optimizer, scheduler_euc,
                             scheduler_hyp, checkpoint_dir, global_step)
    else:
        save_checkpoint_euc(accelerator, decoder, optimizer, scheduler_euc,
                            checkpoint_dir, global_step)


def main() -> None:
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("HELM_TOKENIZER", "meta-llama/Llama-3.1-8B"),
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    train(args, tokenizer)


if __name__ == "__main__":
    main()
