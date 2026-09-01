import argparse
import sys

from .utils import add_flags_from_config

config_args = {
    'training_config': {
        'train': (True, 'If true, MiCE model will return information for load balancing'),
        'min_lr_ratio': (0.1, 'ratio between final target learning rate and initial learning rate'),
        'warm_up_ratio': (0.03, 'percent of steps to use as warm up'),
        'seed': (1234, 'random seed'),
        'lr': (4e-4, 'initial learning rate'),
        'weight_decay': (0.01, 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'packing_ratio': (3.0, 'how many samples to pack into one bin for sample packing'),
        'gradient_accumulation_steps': (256, 'how many steps to update gradients for accelerator'),
        'CHECKPOINT_DIR': ('../ckpt', 'where to save the model'),
        'log_dir': ('../log', 'where to log training dynamics'),
        'data_path': ('../data', 'path to data'),
        'model_name': ('HELM_MiCE', 'One of HELM_D or HELM_MiCE'),
        # Must stay True. DDP's unused-parameter search is a real throughput
        # cost, but HELM genuinely has parameters that receive no gradient on a
        # given step, and DDP errors out without it:
        #   * MoE routing leaves some experts unselected in any given
        #     micro-batch -- measured: 6 of 8 experts starved even at 1024
        #     tokens, because an untrained router concentrates on a few;
        #   * LorentzEmbeddings.add_pos is an LResNet used only when
        #     posit_embed=True, so with HELM's posit_embed=False it is dead
        #     weight that never receives a gradient at any batch size.
        # An earlier version of this port defaulted it to False on the mistaken
        # belief that freezing the attention bias removed the only such
        # parameter. See CALM/experiments/test_helm_calm.py, which caught it.
        'find_unused_parameters': (True, 'whether the accelerator should find unused parameters'),
        'max_batch_size': (1, 'Maximum batch size'),
        'max_seq_len': (2048, 'Maximum sequence length'),
        'project_emb': (False, 'If true, the model will map tokens to space-like dimension of Lorentz vectors'),
        'vocab_size': (128256, 'Vocabulary size of the tokenizer')
    },
    'optimization_config': {
        'attn_impl': ('flash', "'flash' fuses the hyperbolic scores into scaled_dot_product_attention; 'naive' is the literal published formulation"),
        'rope_impl': ('auto', "'auto' picks complex for eager and real for torch.compile; or force 'complex'/'real'"),
        'ce_chunk_size': (512, 'tokens per block in the fused head; peak logit memory is ce_chunk_size x vocab x 4 bytes'),
        'fuse_experts': (True, 'fuse the SwiGLU gate/up projections of each expert into one GEMM'),
        'fuse_residual': (True, 'use the fused Lorentz residual instead of LResNet'),
        'grad_checkpoint': (False, 'recompute block activations in the backward pass to trade compute for memory'),
        'compile': (False, 'wrap the model in torch.compile'),
        'balance_update': (True, 'apply the auxiliary-loss-free routing-bias update each optimizer step'),
    },
    'model_config':{
        'dim': (910, 'Model dimension'),
        'inter_dim': (3640, 'Intermediate dimension for MLP layers'),
        # The model reads `moe_inter_dim`; the released config only defined
        # `mice_inter_dim`, so building HELM-MiCE from its own config raised
        # AttributeError and every example/train_mice_*.sh crashed on startup.
        # Both names are accepted now and reconciled in HelmArgumentParser.
        'mice_inter_dim': (1820, 'Intermediate dimension for MiCE layers'),
        'moe_inter_dim': (None, 'Alias for mice_inter_dim; defaults to it when unset'),
        'n_layers': (16, 'Number of transformer layers'),
        'n_dense_layers': (1, 'Number of dense layers in the model'),
        'n_heads': (14, 'Number of attention heads'),
        # mice
        'n_routed_experts':(8, 'Number of routed experts for MiCE layers'),
        'n_shared_experts':(1, 'Number of shared experts for MiCE layers'),
        'n_activated_experts': (2, 'Number of activated experts in MiCE layers'),
        'n_expert_groups': (1, 'Number of expert groups'),
        'n_limited_groups':(1, 'Number of limited groups for MMiCEoE routing'),
        'score_func': ('softmax', 'Scoring function for MiCE routing'),
        'route_scale':(1., 'Scaling factor for routing scores'),
        'bias_update_speed':(0.005, 'How much to update the bias for gating to ensure expert load balancing'),
        'seq_bal_alpha': (1e-4, 'Scaling for sequence load balancing loss'),
        'train_curv': (True, 'If true, sets the curvatures of the experts as trainable'),
        # hmla
        'q_lora_rank': (0, 'LoRA rank for query projections'),
        'kv_lora_rank': (257, 'LoRA rank for key-value projections'),
        'qk_nope_head_dim': (65, 'Dimension for query-key projections without positional embeddings'),
        'qk_rope_head_dim': (65, 'Dimension for query-key projections with rotary embedding'),
        'v_head_dim':(65, 'Dimension for value projections'),
        # yarn
        'original_seq_len': (2048, 'Original sequence length'),
        'rope_theta': (10000, 'Base for rotary positional encoding'),
        'rope_factor': (40, 'Scaling factor for extended sequence length'),
        'beta_fast': (32, 'Fast beta correction factor'),
        'beta_slow': (1, 'Slow beta correction factor'),
        #helm-d
        'arch': ('L6_W390_A6', 'model architecture for HELM-D, given by La_Wb_Ac, where a is number of layers, b is model dimension, and c is number of heads')
    }
}

def reconcile(args):
    """Fill in derived/aliased options after parsing."""
    if getattr(args, "moe_inter_dim", None) is None:
        args.moe_inter_dim = args.mice_inter_dim
    elif "--mice_inter_dim" not in " ".join(sys.argv):
        args.mice_inter_dim = args.moe_inter_dim
    return args


class HelmArgumentParser(argparse.ArgumentParser):
    """Parser that reconciles the mice_inter_dim / moe_inter_dim split."""

    def parse_args(self, args=None, namespace=None):
        return reconcile(super().parse_args(args, namespace))

    def parse_known_args(self, args=None, namespace=None):
        parsed, extras = super().parse_known_args(args, namespace)
        return reconcile(parsed), extras


parser = HelmArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)