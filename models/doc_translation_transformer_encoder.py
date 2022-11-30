from typing import Optional

from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import PositionalEmbedding
import torch


class DocTranslationTransformerEncoder(TransformerEncoder):
    """TransformerEncoder + functions relevant for doc-level MT."""

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        if self.embed_positions is not None:
            assert not args.encoder_learned_pos
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions * (args.context_size + 1),
                embed_tokens.embedding_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )

    def forward_embedding(self, src_tokens, token_embedding: Optional[torch.Tensor] = None):
        """
        The same as the superclass except for allowing more positional embedding flexibility and
        reinterpreting token_embedding as positional embeddings.
        """
        pos_embeddings = token_embedding
        token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            if pos_embeddings is None:
                pos_embeddings = self.embed_positions(src_tokens)
            x = embed + pos_embeddings
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,  # (B, T_ss)
        src_lengths,  # (B,)
        context_tgt_lens=None,
        true_tgt_lens=None,
        src_context=None,  # (B, C, T_sc)
        tgt_context=None,  # (B, C, T_tc)
        **kwargs
    ):
        assert "token_embeddings" not in kwargs
        return super().forward(src_tokens, src_lengths, **kwargs)
