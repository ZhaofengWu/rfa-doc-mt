import logging
import os

from fairseq import metrics, tokenizer, utils
from fairseq.tasks import TASK_CLASS_NAMES, TASK_REGISTRY, register_task
from fairseq.tasks.translation import TranslationTask

from data_processors.doc_translation_batched_dataset import DocTranslationBatchedDataset
from data_processors.doc_translation_consistency_dataset import (
    DocTranslationConsistencyDataset,
    VALID_PHENOMENA,
    TEST_PHENOMENA,
    compute_consistency,
)
from data_processors.doc_translation_dataset import DocTranslationDataset
from data_processors.doc_translation_dictionary import DocTranslationDictionary
from data_processors.doc_translation_seq2seq_dataset import DocTranslationSeq2seqDataset
from data_processors.doc_translation_sharded_dataset import DocTranslationShardedDataset
from doc_translation_sequence_generator import DocTranslationSequenceGenerator
from utils.array_meter import ArrayMeter

logger = logging.getLogger(__name__)

# Multiple imports cause the same task to be added multiple times, so we need to unregister first.
TASK_REGISTRY.pop("doc_translation", None)
TASK_CLASS_NAMES.discard("DocTranslationTask")


@register_task("doc_translation")
class DocTranslationTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        assert not getattr(args, "print_alignment", False)
        assert src_dict.sep() == tgt_dict.sep()
        assert src_dict.eos() == tgt_dict.eos()
        if args.extra_padding:
            assert args.context_size > 0
            assert args.full_supervision
        if args.seq2seq:
            assert args.use_sep
            assert not args.full_supervision
            assert not args.extra_padding
        assert args.num_saves_per_epoch >= 1
        if args.num_saves_per_epoch > 1:
            assert args.save_interval_updates == 0

        self.full_supervision = args.full_supervision
        self.use_sep = args.use_sep
        self.seq2seq = args.seq2seq
        self.is_test_mode = args.optimizer is None  # TODO: is there a more elegant/reliable check?
        self.valid_consistency = args.valid_subset.endswith("_consistency")
        if self.valid_consistency:
            self.args.eval_bleu = False
            self.is_test_mode = getattr(args, "path", None) is not None
        if args.seq2seq and args.context_size not in {0, 3}:
            assert self.valid_consistency

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)

        parser.add_argument(
            "--context-size",
            type=int,
            default=0,
            help="Number of preceding sentences to use as context. This should really be called"
            " left-context-size, but we didn't for backward compatibility.",
        )
        parser.add_argument(
            "--use-sep",
            action="store_true",
            help="Whether to add SEP after each context sentence.",
        )
        parser.add_argument(
            "--full-supervision",
            action="store_true",
            help="Whether to receive training signal on all targets incuding the context.",
        )
        parser.add_argument(
            "--extra-padding",
            action="store_true",
            help="Full supervision technically can be evaluated with any number of context"
            " sentences, except that the padding is a bit different. This flag"
            " makes the padding consistent between context combinations with the same sum.",
        )
        parser.add_argument(
            "--seq2seq",
            action="store_true",
            help="Whether or not to translate the entire doc at once as a single sequence.",
        )
        parser.add_argument(
            "--num-saves-per-epoch", type=int, default=1, help="Number of saves per epoch."
        )
        parser.add_argument(
            "--doc-bleu",
            default=False,
            action="store_true",
            help="Whether or not to evaluate BLEU on the document level.",
        )
        parser.add_argument(
            "--add-timer",
            default=False,
            action="store_true",
            help="Whether or not to insert timers for the forward pass.",
        )
        parser.add_argument(
            "--duplicate-input-times",
            type=int,
            default=1,
            help="The number of times to duplicate input to simulate a larger batch size.",
        )
        parser.add_argument(
            "--num-decode-examples",
            type=int,
            default=None,
            help="The number of examples to decode.",
        )

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """"Mostly copied from fairseq."""
        d = DocTranslationDictionary()
        for filename in filenames:
            DocTranslationDictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def load_dictionary(cls, filename):
        return DocTranslationDictionary.load(filename)

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        generator = super().build_generator(
            models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )
        return (
            DocTranslationSequenceGenerator(
                generator,
                self.use_sep,
                self.tgt_dict,
                getattr(args, "doc_bleu", False),
                getattr(args, "duplicate_input_times", False),
            )
            if self.is_test_mode and not self.seq2seq
            else generator
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_consistency_set = split.endswith("_consistency")

        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)
        if is_consistency_set:
            self.datasets[split].shuffle = False

        # <copied_from_fairseq>
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        # </copied_from_fairseq>

        if is_consistency_set:
            self.datasets[split] = DocTranslationConsistencyDataset(
                self.datasets[split], self.args.context_size + 1, self.use_sep
            )
        else:
            doc_end_indices = self._load_doc_end_indices(
                os.path.join(data_path, f"{split}.doc_end_indices")
            )
            if self.seq2seq:
                # If context size is 0, we can stick with the sentence level dataset
                if self.args.context_size > 0:
                    self.datasets[split] = DocTranslationSeq2seqDataset(
                        self.datasets[split], doc_end_indices, self.args.context_size + 1
                    )
                elif self.is_test_mode and not self.valid_consistency:
                    self.datasets[split] = DocTranslationBatchedDataset(
                        self.datasets[split], doc_end_indices
                    )
            elif not self.is_test_mode:
                self.datasets[split] = DocTranslationShardedDataset(
                    self.datasets[split],
                    doc_end_indices,
                    self.args.context_size,
                    self.use_sep,
                    self.full_supervision,
                    self.args.extra_padding,
                )
            else:
                self.datasets[split] = DocTranslationDataset(self.datasets[split], doc_end_indices)

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        # This is for fairseq-interactive, which we do not support
        raise NotImplementedError

    def _load_doc_end_indices(self, path):
        with open(path) as f:
            indices = [int(line.strip()) for line in f]
            return indices

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        # <changed_from_fairseq>
        prefix_tokens = None
        if not self.seq2seq:
            prefix_tokens = sample["prefix_tokens"]
            context_tgt_lens = sample["net_input"]["context_tgt_lens"]
            true_tgt_lens = sample["net_input"]["true_tgt_lens"]
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=prefix_tokens)
        # </changed_from_fairseq>
        hyps, refs = [], []
        for i in range(len(gen_out)):
            # <changed_from_fairseq>
            tokens = gen_out[i][0]["tokens"]
            if not self.seq2seq:
                tokens = tokens[context_tgt_lens[i] :]
            hyps.append(decode(tokens))
            target = sample["target"][i]
            if self.full_supervision:
                target = target[context_tgt_lens[i] : context_tgt_lens[i] + true_tgt_lens[i]]
            # </changed_from_fairseq>
            refs.append(
                decode(
                    utils.strip_pad(target, self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.valid_consistency:
            # TODO: it's a bit wasteful that we need to do forward again, but there seems to be no
            # better way w/o significant hacking
            nll_loss = criterion(model, sample, reduce=False)[2]["nll_loss"].cpu()
            assert (nll_loss.sum() - logging_output["nll_loss"]).abs() / nll_loss.numel() < 1e-3
            nll_loss = nll_loss.reshape(logging_output["nsentences"], -1)
            sent_ntokens = (nll_loss > 0).sum(-1)
            assert sent_ntokens.sum() == logging_output["ntokens"] and (sent_ntokens > 0).all()
            nll_loss = nll_loss.sum(-1) / sent_ntokens
            logging_output["_consistency_nll_loss"] = nll_loss.numpy()

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.valid_consistency and "_consistency_nll_loss" in logging_outputs[0]:
            if not self.is_test_mode:
                assert len(logging_outputs) == 1

            for logging_output in logging_outputs:
                metrics.log_custom(
                    lambda: ArrayMeter(),
                    "_consistency_nll_loss",
                    logging_output["_consistency_nll_loss"],
                )

            phenomena = VALID_PHENOMENA if not self.is_test_mode else TEST_PHENOMENA
            split = "valid" if not self.is_test_mode else "test"
            for phenomenon in phenomena:
                metrics.log_derived(
                    f"consistency_{phenomenon}",
                    lambda meters, p=phenomenon: compute_consistency(  # variable capture
                        meters["_consistency_nll_loss"].array, p, split
                    ),
                )

            # TODO: We can't have a derived metric of derived metrics, so we have to re-compute
            # consistency. Though it shouldn't be too slow.
            metrics.log_derived(
                f"consistency",
                lambda meters: sum(
                    compute_consistency(
                        meters["_consistency_nll_loss"].array, phenomenon, split
                    )
                    for phenomenon in phenomena
                )
                / len(phenomena),
            )
