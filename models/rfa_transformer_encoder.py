from typing import List, NamedTuple

from fairseq.models.fairseq_encoder import EncoderOut
import torch
from torch import Tensor

from models.doc_translation_transformer_encoder import DocTranslationTransformerEncoder


class RFACrossAttentionState(NamedTuple):
    s: Tensor
    z: Tensor
    random_matrices: List[Tensor]


RFAEncoderOut = NamedTuple(
    "RFAEncoderOut",
    [(field, EncoderOut._field_types[field]) for field in EncoderOut._fields]
    + [("rfa_cross_attn_state", List[RFACrossAttentionState])],
)


def split_rfa_states(cross_attn_state: RFACrossAttentionState, batch_size):
    context_cross_attn_state = RFACrossAttentionState(
        cross_attn_state.s[:-batch_size],
        cross_attn_state.z[:-batch_size],
        cross_attn_state.random_matrices,
    )
    query_cross_attn_state = RFACrossAttentionState(
        cross_attn_state.s[-batch_size:],
        cross_attn_state.z[-batch_size:],
        cross_attn_state.random_matrices,
    )
    return context_cross_attn_state, query_cross_attn_state


class RFATransformerEncoder(DocTranslationTransformerEncoder):
    def __init__(self, args, *more_args, **kwargs):
        super().__init__(args, *more_args, **kwargs)

        from models.cross_attention import CrossAttentionProject  # avoid circular import

        self.cross_attention_project = CrossAttentionProject(args)

    def forward(self, src_tokens, *args, **kwargs):
        encoder_out = super().forward(src_tokens, *args, **kwargs)
        sep_indices = None
        cross_attn_state = self.cross_attention_project(
            encoder_output=encoder_out.encoder_out,
            key_padding_mask=encoder_out.encoder_padding_mask,
            sep_indices=sep_indices,
        )
        return RFAEncoderOut(
            **{**encoder_out._asdict(), **{"rfa_cross_attn_state": cross_attn_state}}
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: RFAEncoderOut, new_order):
        new_encoder_out = super().reorder_encoder_out(encoder_out, new_order)

        def index_select(t):
            return t.index_select(0, new_order)

        new_cross_attn_state = [
            RFACrossAttentionState(
                index_select(state.s), index_select(state.z), state.random_matrices
            )
            for state in encoder_out.rfa_cross_attn_state
        ]

        return type(encoder_out)(
            **{**new_encoder_out._asdict(), **{"rfa_cross_attn_state": new_cross_attn_state}}
        )
