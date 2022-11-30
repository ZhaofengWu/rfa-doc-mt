from typing import Dict, Optional

import torch
from torch import nn, Tensor

from models.doc_translation_transformer_decoder import DocTranslationTransformerDecoder
from models.dynamic_wrapper import DynamicWrapper
from models.rfa_transformer_decoder_layer import RFATransformerDecoderLayer
from models.rfa_transformer_encoder import RFAEncoderOut
from models.utils import padded_nonzero
from utils.rfa_utils import load_random_matrices, sample_random_matrices


class RFATransformerDecoder(DocTranslationTransformerDecoder):
    def __init__(self, args, *more_args, **kwargs):
        super().__init__(args, *more_args, **kwargs)

        assert not args.cross_self_attention

        self.num_layers = args.decoder_layers
        self.num_heads = args.decoder_attention_heads
        self.head_dim = args.decoder_embed_dim // self.num_heads
        self.proj_dim = args.causal_proj_dim
        self.tau = args.causal_tau
        self.use_sentential_gate = args.use_sentential_gate
        self.decay_sentential_gate = args.decay_sentential_gate
        self.random_matrices = load_random_matrices(
            random_matrices_path=args.random_matrices_path,
            head_dim=self.head_dim,
            proj_dim=self.proj_dim,
            dtype=torch.float16 if args.fp16 else torch.float32,
        )
        self.random_matrices_eval = sample_random_matrices(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            random_matrices=self.random_matrices,
            is_training=False,
        )
        self.random_matrices_eval = nn.Parameter(self.random_matrices_eval)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = RFATransformerDecoderLayer(args, no_encoder_attn=no_encoder_attn)
        call_fn = lambda module, dynamic_args, x, *args, **kwargs: module(
            x, *dynamic_args, *args, **kwargs
        )
        return DynamicWrapper(layer, call_fn)

    def extract_features(self, *args, **kwargs):
        # Basically the same as the superclass implementation, but clearly distinguish args/kwargs
        return self.extract_features_scriptable(*args, **kwargs)

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[RFAEncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **kwargs
    ):
        random_matrices = None
        if self.training:
            random_matrices = sample_random_matrices(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                random_matrices=self.random_matrices,
                is_training=True,
            )

        sep_indices = None
        if self.use_sentential_gate or self.decay_sentential_gate:
            tokens = prev_output_tokens if incremental_state is None else prev_output_tokens[:, -1:]
            sep_indices = padded_nonzero(tokens == self.dictionary.sep())

        for i, layer in enumerate(self.layers):
            rfa_cross_attn_state = None
            if encoder_out is not None and encoder_out.rfa_cross_attn_state is not None:
                rfa_cross_attn_state = encoder_out.rfa_cross_attn_state[i]
            layer.dynamic_args = [
                random_matrices[i] if self.training else self.random_matrices_eval,
                rfa_cross_attn_state,
                sep_indices,
            ]

        out = super().extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **kwargs
        )
        for layer in self.layers:
            layer.reset_dynamic_args()
        return out

    def reorder_incremental_state_scripting(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Same as the superclass, but ignores DynamicWrapper."""
        for module in self.modules():
            if hasattr(module, "reorder_incremental_state") and not isinstance(
                module, DynamicWrapper
            ):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result
