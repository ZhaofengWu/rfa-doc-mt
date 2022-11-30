from fairseq.modules import TransformerDecoderLayer

from models.causal_attention import CausalAttention
from models.cross_attention import CrossAttention
from models.dynamic_wrapper import DynamicWrapper


class RFATransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, *more_args, **kwargs):
        self.embed_dim = args.decoder_embed_dim
        self.num_heads = args.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.tau = args.causal_tau
        self.reparam_proj = args.reparam_proj

        super().__init__(args, *more_args, **kwargs)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        module = CausalAttention(
            args=args,
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tau=self.tau,
            reparam_proj=self.reparam_proj,
        )
        call_fn = lambda module, dynamic_args, query, key, value, **kwargs: (
            module(query, *dynamic_args, **kwargs),
            None,  # attn state
        )
        return DynamicWrapper(module, call_fn)

    def build_encoder_attention(self, embed_dim, args):
        module = CrossAttention(
            args=args,
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tau=self.tau,
            reparam_proj=self.reparam_proj,
        )
        call_fn = lambda module, dynamic_args, query, key, value, **kwargs: (
            module(query, *dynamic_args, incremental_state=kwargs.get("incremental_state")),
            None,  # attn state
        )
        return DynamicWrapper(module, call_fn)

    def forward(self, x, random_matrices, rfa_cross_attn_state, sep_indices, *args, **kwargs):
        self.self_attn.dynamic_args = [random_matrices, sep_indices]
        self.encoder_attn.dynamic_args = [rfa_cross_attn_state, sep_indices]
        out = super().forward(x, *args, **kwargs)
        self.self_attn.reset_dynamic_args()
        self.encoder_attn.reset_dynamic_args()
        return out
