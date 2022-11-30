"""Functions from fairseq whose internals we must change."""
import ast
import collections
import logging
import math
import os
from itertools import chain

from fairseq import checkpoint_utils, metrics, scoring, tasks, utils
from fairseq.checkpoint_utils import checkpoint_paths
from fairseq.data import data_utils
from fairseq.file_io import PathManager
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.tasks.translation import EVAL_BLEU_ORDER, TranslationTask
import fairseq_cli
from fairseq_cli.generate import get_symbols_to_strip_from_output
from fairseq_cli.train import should_stop_early, validate
import numpy as np
import torch


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """
    From https://github.com/pytorch/fairseq/blob/v0.10.0/fairseq/data/language_pair_dataset.py#L16
    except we remove the sorting logic and alignment/constraints processing as well as some
    dim checking.
    """
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        dim = samples[0][key].dim()
        assert all(s[key].dim() == dim for s in samples)
        assert dim == 1
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"] if pad_to_length is not None else None,
        )
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"] if pad_to_length is not None else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


def _div_ceil(a, b):
    return math.ceil(a / float(b))


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    """
    From https://github.com/pytorch/fairseq/blob/v0.10.0/fairseq_cli/train.py#L239
    Changes annotated inline with <changed_from_fairseq>.
    """
    # <changed_from_fairseq>: precomputation for args.num_saves_per_epoch
    update_freq = (  # from fairseq_cli.train.train
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    assert end_of_epoch or epoch_itr.n % update_freq == 0
    num_updates_per_epoch = _div_ceil(  # from fairseq.data.iterators.GroupedIterator.__init__
        len(epoch_itr), update_freq
    )
    save_every = int(_div_ceil(num_updates_per_epoch, args.num_saves_per_epoch))
    # </changed_from_fairseq>
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
        or num_updates >= max_update
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
        # <changed_from_fairseq>: support args.num_saves_per_epoch
        or (
            args.num_saves_per_epoch > 1
            and num_updates > 0
            # we don't use num_updatets which doesn't reset to 0 every epoch
            and (epoch_itr.n // update_freq) % save_every == 0
            and num_updates >= args.validate_after_updates
        )
        # </changed_from_fairseq>
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or num_updates >= max_update
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    # <changed_from_fairseq>: add end_of_epoch criteron for calling should_stop_early
    should_stop = (
        (end_of_epoch and should_stop_early(args, valid_losses[0]))
        or num_updates >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )
    # </changed_from_fairseq>

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


fairseq_cli.train.validate_and_save = validate_and_save


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    """
    From https://github.com/pytorch/fairseq/blob/v0.10.0/fairseq/checkpoint_utils.py#L23
    Changes annotated inline with <changed_from_fairseq>.
    """
    from fairseq import distributed_utils, meters

    # only one worker should attempt to create the required dir
    if args.distributed_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if args.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if args.no_save:
        return

    trainer.consolidate_optimizer()

    if not trainer.is_data_parallel_master:
        return

    def is_better(a, b):
        return a >= b if args.maximize_best_checkpoint_metric else a <= b

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    suffix = getattr(args, "checkpoint_suffix", "")
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch
        and not args.no_epoch_checkpoints
        and epoch % args.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        not end_of_epoch
        and args.save_interval_updates > 0
        and updates % args.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and args.keep_best_checkpoints > 0:
        # <changed_from_fairseq>: remove the condition to always save ckpts here; increase decimal
        # places
        checkpoint_conds[
            "checkpoint.best_{}_{:.3f}.pt".format(args.best_checkpoint_metric, val_loss)
        ] = True
        # </changed_from_fariseq>
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not args.no_last_checkpoints

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            PathManager.copy(checkpoints[0], cp, overwrite=True)

        write_timer.stop()
        logger.info(
            "saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r"checkpoint_\d+_(\d+)\.pt"
        )
        for old_chk in checkpoints[args.keep_interval_updates :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(args.save_dir, pattern=r"checkpoint(\d+)\.pt")
        for old_chk in checkpoints[args.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            args.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*)\.pt".format(
                args.best_checkpoint_metric
            ),
        )
        if not args.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[args.keep_best_checkpoints :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


checkpoint_utils.save_checkpoint = save_checkpoint


def reduce_metrics(self, logging_outputs, criterion):
    """
    From https://github.com/pytorch/fairseq/blob/v0.10.0/fairseq/tasks/translation.py#L361
    Changes annotated inline with <changed_from_fairseq>.
    """
    # <changed_from_fairseq>: work around hack for super()
    super(TranslationTask, self).reduce_metrics(logging_outputs, criterion)
    # </changed_from_fairseq>
    if self.args.eval_bleu:

        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        counts, totals = [], []
        for i in range(EVAL_BLEU_ORDER):
            counts.append(sum_logs("_bleu_counts_" + str(i)))
            totals.append(sum_logs("_bleu_totals_" + str(i)))

        if max(totals) > 0:
            # log counts as numpy arrays -- log_scalar will sum them correctly
            metrics.log_scalar("_bleu_counts", np.array(counts))
            metrics.log_scalar("_bleu_totals", np.array(totals))
            metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
            metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

            def compute_bleu(meters):
                import inspect
                import sacrebleu

                fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                if "smooth_method" in fn_sig:
                    smooth = {"smooth_method": "exp"}
                else:
                    smooth = {"smooth": "exp"}
                bleu = sacrebleu.compute_bleu(
                    correct=meters["_bleu_counts"].sum,
                    total=meters["_bleu_totals"].sum,
                    sys_len=meters["_bleu_sys_len"].sum,
                    ref_len=meters["_bleu_ref_len"].sum,
                    **smooth
                )
                # <changed_from_fairseq>: 2 -> 3
                return round(bleu.score, 3)
                # </changed_from_fairseq>

            metrics.log_derived("bleu", compute_bleu)


TranslationTask.reduce_metrics = reduce_metrics


def _main(args, output_file):
    """
    From https://github.com/pytorch/fairseq/blob/v0.10.0/fairseq_cli/generate.py#L51
    Changes annotated inline with <changed_from_fairseq> to support doc-level bleu.
    """
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    if args.lm_path is not None:
        overrides["data"] = args.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [args.lm_path],
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({args.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    # <changed_from_fairseq>
    if args.add_timer:
        gen_forward_timer = StopwatchMeter()
        # We could've done this elegantly with hooks, but fairseq calls forward directly...
        orig_encoder_forward_fn = models[0].encoder.forward_torchscript
        def encoder_forward_fn(*args, **kwargs):
            torch.cuda.synchronize()
            gen_forward_timer.start()
            ret = orig_encoder_forward_fn(*args, **kwargs)
            torch.cuda.synchronize()
            gen_forward_timer.stop()
            return ret
        models[0].encoder.forward_torchscript = encoder_forward_fn

        orig_decoder_forward_fn = models[0].decoder.forward
        def decoder_forward_fn(*args, **kwargs):
            torch.cuda.synchronize()
            gen_forward_timer.start()
            ret = orig_decoder_forward_fn(*args, **kwargs)
            torch.cuda.synchronize()
            gen_forward_timer.stop()
            return ret
        models[0].decoder.forward = decoder_forward_fn
    # </changed_from_fairseq>

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight}
    generator = task.build_generator(
        models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(args, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    # <changed_from_fairseq>
    for sample_idx, sample in enumerate(progress):
        # THe 1 warmup batch is in addition to args.num_decode_examples
        if args.num_decode_examples is not None and sample_idx > args.num_decode_examples:
            break
    # </changed_from_fairseq>
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample["target"][:, : args.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        # <changed_from_fairseq>
        if sample_idx >= 1:  # warmup
        # </changed_from_fairseq>
            gen_timer.start()
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        # <changed_from_fairseq>
        if sample_idx >= 1:
            gen_timer.stop(num_generated_tokens)
        elif args.add_timer:
            gen_forward_timer.sum = gen_forward_timer.n = 0
        # </changed_from_fairseq>

        # <changed_from_fairseq>
        if len(hypos) == 1 and len(sample["id"].tolist()) > 1:
            assert (
                sample["target"] is not None and "src_tokens" in sample["net_input"]
                and align_dict is None and src_dict is not None and args.nbest == 1
                and not args.print_alignment and not args.print_step
                and not getattr(args, "retain_iter_history", False)
            )
            src_tokens = torch.cat(
                [
                    utils.strip_pad(tokens, tgt_dict.pad())
                    for tokens in sample["net_input"]["src_tokens"]
                ],
                dim=0,
            )
            target_tokens = torch.cat(
                [
                    utils.strip_pad(tokens, tgt_dict.pad()).int().cpu()
                    for tokens in sample["target"]
                ],
                dim=0,
            )
            src_str = src_dict.string(src_tokens, args.remove_bpe)
            target_str = tgt_dict.string(
                target_tokens,
                args.remove_bpe,
                escape_unk=True,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                    generator
                ),
            )
            src_str = decode_fn(src_str)
            target_str = decode_fn(target_str)
            if not args.quiet:
                print("S-{}\t{}".format(0, src_str), file=output_file)
                print("T-{}\t{}".format(0, target_str), file=output_file)

            hypo = hypos[0][0]
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            detok_hypo_str = decode_fn(hypo_str)
            if not args.quiet:
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print(
                    "H-{}\t{}\t{}".format(0, score, hypo_str),
                    file=output_file,
                )
                # detokenized hypothesis
                print(
                    "D-{}\t{}\t{}".format(0, score, detok_hypo_str),
                    file=output_file,
                )
                print(
                    "P-{}\t{}".format(
                        0,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"]
                                .div_(math.log(2))
                                .tolist(),
                            )
                        ),
                    ),
                    file=output_file,
                )

            if align_dict is not None or args.remove_bpe is not None:
                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                target_tokens = tgt_dict.encode_line(
                    target_str, add_if_not_exist=True
                )
                hypo_tokens = tgt_dict.encode_line(
                    detok_hypo_str, add_if_not_exist=True
                )
            if hasattr(scorer, "add_string"):
                scorer.add_string(target_str, detok_hypo_str)
            else:
                scorer.add(target_tokens, hypo_tokens)
        elif task.args.seq2seq and task.args.context_size == 0:
            assert (
                sample["target"] is not None and "src_tokens" in sample["net_input"]
                and align_dict is None and src_dict is not None and args.nbest == 1
                and not args.print_alignment and not args.print_step
                and not getattr(args, "retain_iter_history", False)
            )
            src_tokens = sample["net_input"]["src_tokens"]
            target = sample["target"]
            assert len(src_tokens) == len(target) == len(hypos)
            src_tokens = src_tokens.reshape(-1, 4, src_tokens.shape[-1])
            target = target.reshape(-1, 4, target.shape[-1])
            hypos = [hypos[i * 4 : i * 4 + 4] for i in range(len(hypos) // 4)]
            for src, tgt, hyp in zip(src_tokens, target, hypos):
                src_tokens = torch.cat(
                    [utils.strip_pad(tokens, tgt_dict.pad()) for tokens in src], dim=0
                )
                target_tokens = torch.cat(
                    [utils.strip_pad(tokens, tgt_dict.pad()).int().cpu() for tokens in tgt], dim=0
                )
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                target_str = tgt_dict.string(
                    target_tokens,
                    args.remove_bpe,
                    escape_unk=True,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                        generator
                    ),
                )
                src_str = decode_fn(src_str)
                target_str = decode_fn(target_str)
                if not args.quiet:
                    print("S-{}\t{}".format(0, src_str), file=output_file)
                    print("T-{}\t{}".format(0, target_str), file=output_file)

                hypo = {
                    "tokens": torch.cat([h[0]["tokens"] for h in hyp], dim=0),
                    "alignment": torch.FloatTensor([]),
                    "score": hyp[0][0]["score"],
                    "positional_scores": torch.cat([h[0]["positional_scores"] for h in hyp], dim=0),
                }
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not args.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(0, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(0, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            0,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                if align_dict is not None or args.remove_bpe is not None:
                    # Convert back to tokens for evaluation with unk replacement and/or without BPE
                    target_tokens = tgt_dict.encode_line(
                        target_str, add_if_not_exist=True
                    )
                    hypo_tokens = tgt_dict.encode_line(
                        detok_hypo_str, add_if_not_exist=True
                    )
                if hasattr(scorer, "add_string"):
                    scorer.add_string(target_str, detok_hypo_str)
                else:
                    scorer.add(target_tokens, hypo_tokens)
        else:
        # </changed_from_fairseq>
            for i, sample_id in enumerate(sample["id"].tolist()):
                has_target = sample["target"] is not None

                # Remove padding
                if "src_tokens" in sample["net_input"]:
                    src_tokens = utils.strip_pad(
                        sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                    )
                else:
                    src_tokens = None

                target_tokens = None
                if has_target:
                    target_tokens = (
                        utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                    )

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(
                        sample_id
                    )
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(
                            target_tokens,
                            args.remove_bpe,
                            escape_unk=True,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                                generator
                            ),
                        )

                src_str = decode_fn(src_str)
                if has_target:
                    target_str = decode_fn(target_str)

                if not args.quiet:
                    if src_dict is not None:
                        print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                    if has_target:
                        print("T-{}\t{}".format(sample_id, target_str), file=output_file)

                # Process top predictions
                for j, hypo in enumerate(hypos[i][: args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    if not args.quiet:
                        score = hypo["score"] / math.log(2)  # convert to base 2
                        # original hypothesis (after tokenization and BPE)
                        print(
                            "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                            file=output_file,
                        )
                        # detokenized hypothesis
                        print(
                            "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                            file=output_file,
                        )
                        print(
                            "P-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    map(
                                        lambda x: "{:.4f}".format(x),
                                        # convert from base e to base 2
                                        hypo["positional_scores"]
                                        .div_(math.log(2))
                                        .tolist(),
                                    )
                                ),
                            ),
                            file=output_file,
                        )

                        if args.print_alignment:
                            print(
                                "A-{}\t{}".format(
                                    sample_id,
                                    " ".join(
                                        [
                                            "{}-{}".format(src_idx, tgt_idx)
                                            for src_idx, tgt_idx in alignment
                                        ]
                                    ),
                                ),
                                file=output_file,
                            )

                        if args.print_step:
                            print(
                                "I-{}\t{}".format(sample_id, hypo["steps"]),
                                file=output_file,
                            )

                        if getattr(args, "retain_iter_history", False):
                            for step, h in enumerate(hypo["history"]):
                                _, h_str, _ = utils.post_process_prediction(
                                    hypo_tokens=h["tokens"].int().cpu(),
                                    src_str=src_str,
                                    alignment=None,
                                    align_dict=None,
                                    tgt_dict=tgt_dict,
                                    remove_bpe=None,
                                )
                                print(
                                    "E-{}_{}\t{}".format(sample_id, step, h_str),
                                    file=output_file,
                                )

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(
                                target_str, add_if_not_exist=True
                            )
                            hypo_tokens = tgt_dict.encode_line(
                                detok_hypo_str, add_if_not_exist=True
                            )
                        if hasattr(scorer, "add_string"):
                            scorer.add_string(target_str, detok_hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    # <changed_from_fairseq>
    if args.add_timer:
        gen_forward_timer.n = gen_timer.n
        logger.info("Forward time only: {:.2f} tokens/s".format(1.0 / gen_forward_timer.avg))
    # </changed_from_fairseq>
    if has_target:
        if args.bpe and not args.sacrebleu:
            if args.remove_bpe:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                args.gen_subset, args.beam, scorer.result_string()
            ),
            file=output_file,
        )

    return scorer


fairseq_cli.generate._main = _main
