from .data_processors.doc_translation_task import DocTranslationTask
import fairseq_adaptations  # import early to ensure all injections happen first
from .models.rfa_transformer import rfa_transformer_iwslt_de_en
