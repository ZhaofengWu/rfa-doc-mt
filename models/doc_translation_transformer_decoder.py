from fairseq.modules import PositionalEmbedding
from fairseq.models.transformer import TransformerDecoder


class DocTranslationTransformerDecoder(TransformerDecoder):
    """TransformerDecoder + functions relevant for doc-level MT."""

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)

        if self.embed_positions is not None:
            assert not args.decoder_learned_pos
            self.embed_positions = PositionalEmbedding(
                args.max_target_positions * (args.context_size + 1),
                embed_tokens.embedding_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
