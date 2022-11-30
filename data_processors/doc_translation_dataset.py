from fairseq.data import FairseqDataset, LanguagePairDataset
import numpy as np
import torch

from data_processors.utils import pad_and_stack


class DocTranslationDataset(FairseqDataset):
    def __init__(self, sent_level_dataset: LanguagePairDataset, doc_end_indices):
        # Simplifications
        assert sent_level_dataset.buckets is None
        assert sent_level_dataset.align_dataset is None
        assert sent_level_dataset.constraints is None
        assert sent_level_dataset.src_lang_id is None
        assert sent_level_dataset.tgt_lang_id is None

        # Checks
        assert doc_end_indices[-1] == len(sent_level_dataset) - 1

        self.sent_level_dataset = sent_level_dataset
        doc_start_indices = [0] + [i + 1 for i in doc_end_indices[:-1]]
        self.doc_indices = [
            list(range(start, end + 1)) for start, end in zip(doc_start_indices, doc_end_indices)
        ]
        assert all(len(indices) > 0 for indices in self.doc_indices)
        assert (sent_level_dataset.tgt is None) == (sent_level_dataset.tgt_sizes is None)
        self.has_tgt = sent_level_dataset.tgt is not None
        assert self.has_tgt  # we should actually always have targets in our use cases

        self.pad_idx = self.sent_level_dataset.src_dict.pad()

    def __getitem__(self, index, strip_eos=True):
        sent_pairs = [self.sent_level_dataset[i] for i in self.doc_indices[index]]
        # Remove eos from all sources which will be added back in the generator as either sep or eos
        for sent_pair in sent_pairs:
            assert sent_pair["source"][-1].item() == self.sent_level_dataset.eos
            if strip_eos:
                sent_pair["source"] = sent_pair["source"][:-1]

        return {
            "id": index,
            "source": pad_and_stack(
                [sent_pair["source"] for sent_pair in sent_pairs], self.pad_idx
            ),
            "target": pad_and_stack([sent_pair["target"] for sent_pair in sent_pairs], self.pad_idx)
            if self.has_tgt
            else None,
        }

    def __len__(self):
        return len(self.doc_indices)

    def collater(self, samples, pad_to_length=None):
        assert len(samples) == 1
        sample = samples[0]
        nsentences = len(sample["source"])
        batch = {
            "id": torch.LongTensor([sample["id"]] * nsentences),
            "nsentences": nsentences,
            "ntokens": (sample["target" if self.has_tgt else "source"] != self.pad_idx).sum(),
            "net_input": {
                "src_tokens": sample["source"],
            },
            "target": sample["target"],
            "pad_idx": self.pad_idx,
        }
        return batch

    def num_tokens(self, index):
        raise NotImplementedError

    def size(self, index):
        raise NotImplementedError

    def ordered_indices(self):
        if self.sent_level_dataset.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices

    def filter_indices_by_size(self, indices, max_sizes):
        return indices, []

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        # One document per batch
        return [[i] for i in indices]
