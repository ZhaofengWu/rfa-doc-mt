from fairseq.data import FairseqDataset, LanguagePairDataset
import numpy as np
import torch

from data_processors.utils import pad_and_stack, get_sent_pairs
from fairseq_adaptations import collate


class DocTranslationShardedDataset(FairseqDataset):
    def __init__(
        self,
        sent_level_dataset: LanguagePairDataset,
        doc_end_indices,
        left_context_size,
        use_sep,
        full_supervision,
        extra_padding,
    ):
        # Simplifications
        assert sent_level_dataset.buckets is None
        assert sent_level_dataset.align_dataset is None
        assert sent_level_dataset.constraints is None
        assert sent_level_dataset.src_lang_id is None
        assert sent_level_dataset.tgt_lang_id is None
        assert not sent_level_dataset.append_eos_to_target
        assert not sent_level_dataset.append_bos
        assert not sent_level_dataset.remove_eos_from_source
        assert not sent_level_dataset.left_pad_source
        assert not sent_level_dataset.left_pad_target

        # Checks
        assert doc_end_indices[-1] == len(sent_level_dataset) - 1

        self.sent_level_dataset = sent_level_dataset
        doc_start_indices = [0] + [i + 1 for i in doc_end_indices[:-1]]
        self.doc_start_indices = doc_start_indices
        self.doc_end_indices = doc_end_indices
        self.doc_indices = [
            list(range(start, end + 1)) for start, end in zip(doc_start_indices, doc_end_indices)
        ]
        assert all(len(indices) > 0 for indices in self.doc_indices)

        self.left_context_size = left_context_size
        self.use_sep = use_sep
        self.full_supervision = full_supervision
        self.extra_padding = extra_padding

        self.sentidx2docidx = self._compute_sentidx2docidx()
        if extra_padding:
            self.shard_info = self._compute_shard_info()
            assert len(self) == len(self.shard_info)
        else:
            assert len(self) == len(sent_level_dataset)

    def _compute_sentidx2docidx(self):
        sentidx2docidx = []
        for i, indices in enumerate(self.doc_indices):
            sentidx2docidx.extend([i] * len(indices))
        return sentidx2docidx

    def _compute_shard_info(self):
        shard_info = []  # list of (sent_idx, left_context_adjustment)
        for indices in self.doc_indices:
            sent_indices = indices + [indices[-1]] * self.left_context_size
            left_context_adj = [0] * len(sent_indices)
            left_context_adj[-self.left_context_size :] = range(1, self.left_context_size + 1)
            shard_info.extend(zip(sent_indices, left_context_adj))
        return shard_info

    def _get_sent_pairs(self, index):
        left_context_adj = 0
        if self.extra_padding:
            index, left_context_adj = self.shard_info[index]

        doc_idx = self.sentidx2docidx[index]
        doc_start_idx = self.doc_start_indices[doc_idx]
        doc_end_idx = self.doc_end_indices[doc_idx]
        shard_start_idx = max(index - self.left_context_size + left_context_adj, doc_start_idx)
        shard_end_idx = min(index, doc_end_idx)
        tgt_end_idx = index if not self.full_supervision else shard_end_idx
        return (
            get_sent_pairs(
                self.sent_level_dataset,
                shard_start_idx,
                shard_end_idx,
                self.use_sep,
                tgt_end=tgt_end_idx,
            ),
            index - shard_start_idx,
        )

    def __getitem__(self, index):
        (sources, targets), rel_idx = self._get_sent_pairs(index)
        prefix_tokens = torch.cat(targets[:rel_idx], dim=0) if rel_idx > 0 else torch.LongTensor([])
        all_targets = torch.cat(targets, dim=0)
        prev_output_tokens = all_targets.roll(1, dims=0)
        return {
            "id": index,
            "source": torch.cat(sources, dim=0),
            "target": targets[-1] if not self.full_supervision else all_targets,
            "prev_output_tokens": prev_output_tokens,
            "prefix_tokens": prefix_tokens,
            "context_tgt_len": torch.LongTensor([len(prefix_tokens)]),
            "true_tgt_len": torch.LongTensor([len(targets[rel_idx])]),
        }

    def __len__(self):
        return len(self.sentidx2docidx) if not self.extra_padding else len(self.shard_info)

    def collater(self, samples, pad_to_length=None):
        pad_idx = self.sent_level_dataset.src_dict.pad()

        collated_samples = collate(
            samples,
            pad_idx=pad_idx,
            eos_idx=self.sent_level_dataset.eos,
            left_pad_source=self.sent_level_dataset.left_pad_source,
            left_pad_target=self.sent_level_dataset.left_pad_target,
            input_feeding=self.sent_level_dataset.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.sent_level_dataset.pad_to_multiple,
        )

        # TODO: we need to change "num_tokens" to get accurate reduced metrics, but this doesn't
        # affect training
        collated_samples["prefix_tokens"] = pad_and_stack(
            [sample["prefix_tokens"] for sample in samples], pad_idx
        )
        collated_samples["net_input"]["context_tgt_lens"] = torch.cat(
            [sample["context_tgt_len"] for sample in samples], dim=0
        )
        collated_samples["net_input"]["true_tgt_lens"] = torch.cat(
            [sample["true_tgt_len"] for sample in samples], dim=0
        )

        return collated_samples

    def _total_size(self, index):
        e = self[index]
        return (
            len(e["source"]),
            len(e["prev_output_tokens"]) if self.sent_level_dataset.tgt_sizes is not None else 0,
        )

    def num_tokens(self, index):
        return max(*self._total_size(index))

    def size(self, index):
        if self.extra_padding:
            index, _ = self.shard_info[index]
        return self.sent_level_dataset.size(index)

    def ordered_indices(self):
        """Adapted from fairseq"""
        if self.sent_level_dataset.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        src_lens = []
        tgt_lens = []
        for i in range(len(self)):
            src_len, tgt_len = self._total_size(i)
            src_lens.append(src_len)
            tgt_lens.append(tgt_len)
        src_lens = np.array(src_lens)
        tgt_lens = np.array(tgt_lens)

        # sort by target length, then source length
        if self.sent_level_dataset.tgt_sizes is not None:
            indices = indices[np.argsort(tgt_lens[indices], kind="mergesort")]
        return indices[np.argsort(src_lens[indices], kind="mergesort")]

    def filter_indices_by_size(self, indices, max_sizes):
        filtered_indices, ignored = super().filter_indices_by_size(indices, max_sizes)
        if not self.extra_padding:
            filtered_indices2, ignored2 = self.sent_level_dataset.filter_indices_by_size(
                indices, max_sizes
            )
            assert (filtered_indices == filtered_indices2).all()
            assert ignored == ignored2
        return filtered_indices, ignored
