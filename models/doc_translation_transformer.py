from fairseq.models import (
    ARCH_MODEL_REGISTRY,
    MODEL_REGISTRY,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
import torch

from models.doc_translation_transformer_decoder import DocTranslationTransformerDecoder
from models.doc_translation_transformer_encoder import DocTranslationTransformerEncoder

# Multiple imports cause the same model/architecture to be added multiple times,
# so we need to unregister first.
MODEL_REGISTRY.pop("doc_translation_transformer", None)
ARCH_MODEL_REGISTRY.pop("doc_translation_transformer_iwslt_de_en", None)


@register_model("doc_translation_transformer")
class DocTranslationTransformer(TransformerModel):
    """TransformerModel + functions relevant for doc-level MT."""

    def __init__(self, args, *more_args, **kwargs):
        super().__init__(args, *more_args, **kwargs)
        self.pad_idx = self.encoder.embed_tokens.padding_idx
        self.full_supervision = getattr(args, "full_supervision", False)
        self.seq2seq = getattr(args, "seq2seq", False)
        self.valid_consistency = args.valid_subset.endswith("_consistency")

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return DocTranslationTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, src_dict, embed_tokens):
        return DocTranslationTransformerDecoder(args, src_dict, embed_tokens)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        context_tgt_lens=None,
        true_tgt_lens=None,
        return_all_hiddens: bool = True,
        **kwargs,
    ):
        encoder_out = self.encoder(src_tokens, src_lengths, return_all_hiddens=return_all_hiddens)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            **kwargs,
        )
        if not (
            self.full_supervision
            or (self.valid_consistency and not self.training)
            or self.seq2seq
        ):
            assert (
                (context_tgt_lens + true_tgt_lens) == (prev_output_tokens != self.pad_idx).sum(1)
            ).all()
            output = decoder_out[0]  # (b, max(context_tgt_len + true_tgt_len), vocab_size)
            assert len(output) == len(context_tgt_lens)
            rolled_output = torch.stack(
                [
                    single_output.roll(-context_tgt_len.item(), dims=0)
                    for single_output, context_tgt_len in zip(output, context_tgt_lens)
                ],
                dim=0,
            )
            max_true_tgt_len = true_tgt_lens.max()
            decoder_out = (rolled_output[:, :max_true_tgt_len, :],) + decoder_out[1:]
        return decoder_out


@register_model_architecture(
    "doc_translation_transformer", "doc_translation_transformer_iwslt_de_en"
)
def doc_translation_transformer_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
