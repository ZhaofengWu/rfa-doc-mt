from fairseq.models import (
    ARCH_MODEL_REGISTRY,
    MODEL_REGISTRY,
    register_model,
    register_model_architecture,
)

from models.doc_translation_transformer import (
    DocTranslationTransformer,
    doc_translation_transformer_iwslt_de_en,
)
from models.rfa_transformer_decoder import RFATransformerDecoder
from models.rfa_transformer_encoder import RFATransformerEncoder


# TODO: LM support

# Multiple imports cause the same model/architecture to be added multiple times,
# so we need to unregister first.
MODEL_REGISTRY.pop("rfa_transformer", None)
ARCH_MODEL_REGISTRY.pop("rfa_transformer_iwslt_de_en", None)


@register_model("rfa_transformer")
class RFATransformer(DocTranslationTransformer):
    @staticmethod
    def add_args(parser):
        DocTranslationTransformer.add_args(parser)

        # --- RFA arguments, mostly taken from the original repo ---
        parser.add_argument(
            "--random-matrices-path",
            type=str,
            required=True,
            help="Path to pre-generated random matrices.",
        )
        parser.add_argument(
            "--cross-proj-dim",
            type=int,
            default=64,
            metavar="N",
            help="projection size for cross rfa",
        )
        parser.add_argument(
            "--causal-proj-dim",
            type=int,
            default=64,
            metavar="N",
            help="projection size for causal rfa",
        )
        parser.add_argument("--cross-tau", type=float, default=1.0, metavar="D", help="tau for rfa")
        parser.add_argument(
            "--causal-tau", type=float, default=1.0, metavar="D", help="tau for rfa"
        )
        parser.add_argument(
            "--reparam-proj",
            default=False,
            action="store_true",
            help="whether or not to reparameterze random matrices in rfa",
        )
        parser.add_argument(
            "--learned-tau",
            default=False,
            action="store_true",
            help="whether or not to learn tau in rfa",
        )
        parser.add_argument(
            "--norm-rescale",
            default=False,
            action="store_true",
            help="whether or not to rescale keys by their norms",
        )
        parser.add_argument(
            "--cuda-causal-rfa",
            default=False,
            action="store_true",
            help="whether or not to use custom cuda kernel for causal rfa",
        )
        parser.add_argument("--init-scale", type=float, default=1.0, metavar="D", help="init scale")
        parser.add_argument(
            "--random-feature",
            type=str,
            default="rrf",
            choices=["rrf", "prf"],
            help="Random feature type",
        )
        # --- doc translation arguments ---
        parser.add_argument(
            "--use-sentential-gate",
            default=False,
            action="store_true",
            help="Whether or not to enable recency bias between sentences",
        )
        parser.add_argument(
            "--decay-sentential-gate",
            default=False,
            action="store_true",
            help="Whether or not to enable recency bias between sentences in a decaying fashion",
        )
        parser.add_argument(
            "--decay-gate-bias",
            type=float,
            default=0.0,
            help="The initialization bias for the decay sentential gate.",
        )
        parser.add_argument(
            "--cuda-inference",
            default=False,
            action="store_true",
            help="Whether or not to use cuda kernel for inference.",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return RFATransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return RFATransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def load_state_dict(self, state_dict, strict=True, args=None):
        # We allow loading from a pretrained model without a g_proj
        incompatible_keys = super().load_state_dict(state_dict, strict=False, args=args)
        assert len(incompatible_keys.unexpected_keys) == 0
        assert all('.g_proj.' in key for key in incompatible_keys.missing_keys)
        return incompatible_keys._replace(missing_keys=[])

@register_model_architecture("rfa_transformer", "rfa_transformer_iwslt_de_en")
def rfa_transformer_iwslt_de_en(args):
    doc_translation_transformer_iwslt_de_en(args)
