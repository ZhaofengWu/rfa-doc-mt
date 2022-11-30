from fairseq.data import FairseqDataset, LanguagePairDataset
import numpy as np
import torch

from data_processors.utils import get_sent_pairs
from data_processors.doc_translation_consistency_dataset import DOC_LEN


class DocTranslationSeq2seqDataset(FairseqDataset):
    def __init__(
        self,
        sent_level_dataset: LanguagePairDataset,
        doc_end_indices,
        shard_size,
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
        doc_start_indices = [i + (DOC_LEN - shard_size) for i in doc_start_indices]
        self.doc_start_indices = doc_start_indices
        self.doc_end_indices = doc_end_indices
        self.doc_indices = [
            list(range(start, end + 1)) for start, end in zip(doc_start_indices, doc_end_indices)
        ]
        assert all(len(indices) > 0 for indices in self.doc_indices)

        self.src_sizes = np.array(
            [
                sum(self.sent_level_dataset.src_sizes[i] for i in indices)
                for indices in self.doc_indices
            ]
        )
        self.tgt_sizes = np.array(
            [
                sum(self.sent_level_dataset.tgt_sizes[i] for i in indices)
                for indices in self.doc_indices
            ]
        )

    def __getitem__(self, index):
        sources, targets = get_sent_pairs(
            self.sent_level_dataset,
            self.doc_start_indices[index],
            self.doc_end_indices[index],
            True,
        )
        return {
            "id": index,
            "source": torch.cat(sources, dim=0),
            "target": torch.cat(targets, dim=0),
        }

    def __len__(self):
        return len(self.doc_end_indices)

    def collater(self, samples, pad_to_length=None):
        return self.sent_level_dataset.collater(samples, pad_to_length=pad_to_length)

    def num_tokens(self, index):
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        if self.sent_level_dataset.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        # sort by target length, then source length
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    def filter_indices_by_size(self, indices, max_sizes):
        return super().filter_indices_by_size(indices, max_sizes)
