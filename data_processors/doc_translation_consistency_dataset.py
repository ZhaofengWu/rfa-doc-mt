import os
import subprocess
import tempfile

from fairseq.data import FairseqDataset
import numpy as np
import torch

from data_processors.utils import get_sent_pairs
from fairseq_adaptations import collate

# Sorting here is important to match the pre-processing order
VALID_PHENOMENA = sorted(["deixis", "lex_cohesion"])
TEST_PHENOMENA = sorted(["deixis", "lex_cohesion", "ellipsis_infl", "ellipsis_vp"])
VALID_PHENOMENON_BOUNDARY = {"deixis": (0, 1000), "lex_cohesion": (1000, 2124)}
TEST_PHENOMENON_BOUNDARY = {
    "deixis": (0, 5000),
    "ellipsis_infl": (5000, 7520),
    "ellipsis_vp": (7520, 12723),
    "lex_cohesion": (12723, 16151),
}
REPO_PATH = "data/consistency_sets"

DOC_LEN = 4


class DocTranslationConsistencyDataset(FairseqDataset):
    def __init__(self, sent_level_dataset, shard_size, use_sep):
        assert len(sent_level_dataset) % DOC_LEN == 0
        self.sent_level_dataset = sent_level_dataset
        self.shard_size = min(shard_size, DOC_LEN)
        self.use_sep = use_sep

        src_sizes = []
        tgt_sizes = []
        for i in range(len(sent_level_dataset) // DOC_LEN):
            indices = range(DOC_LEN * i + DOC_LEN - self.shard_size, DOC_LEN * i + DOC_LEN)
            src_sizes.append(sum(sent_level_dataset.src_sizes[j] for j in indices))
            tgt_sizes.append(sum(sent_level_dataset.tgt_sizes[j] for j in indices))
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)

    def __getitem__(self, index):
        sources, targets = get_sent_pairs(
            self.sent_level_dataset,
            DOC_LEN * index + DOC_LEN - self.shard_size,
            DOC_LEN * index + DOC_LEN - 1,
            self.use_sep,
        )
        return {
            "id": index,
            "source": torch.cat(sources, dim=0),
            "target": torch.cat(targets, dim=0),
        }

    def __len__(self):
        return len(self.sent_level_dataset) // DOC_LEN

    def collater(self, samples, pad_to_length=None):
        return collate(
            samples,
            pad_idx=self.sent_level_dataset.src_dict.pad(),
            eos_idx=self.sent_level_dataset.eos,
            left_pad_source=self.sent_level_dataset.left_pad_source,
            left_pad_target=self.sent_level_dataset.left_pad_target,
            input_feeding=self.sent_level_dataset.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.sent_level_dataset.pad_to_multiple,
        )

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
        return np.arange(len(self), dtype=np.int64)

    def filter_indices_by_size(self, indices, max_size):
        return indices, []


def compute_consistency(nll_loss, phenomenon, split):
    split = "dev" if split == "valid" else "test"
    phenomenon_boundary = VALID_PHENOMENON_BOUNDARY if split == "dev" else TEST_PHENOMENON_BOUNDARY
    assert phenomenon_boundary is not None
    # fairseq sometimes invoke this before the final aggregation, in which case we do nothing
    if max(boundary[1] for boundary in phenomenon_boundary.values()) != len(nll_loss):
        return -1
    start, end = phenomenon_boundary[phenomenon]
    with tempfile.NamedTemporaryFile("w") as f:
        for loss in nll_loss[start:end]:
            f.write(f"{loss.item()}\n")
        f.flush()
        result = subprocess.run(
            [
                "python",
                os.path.join(REPO_PATH, "scripts/evaluate_consistency.py"),
                "--repo-dir",
                REPO_PATH,
                "--test",
                f"{phenomenon}_{split}" if phenomenon in VALID_PHENOMENA else phenomenon,
                "--scores",
                f.name,
            ],
            capture_output=True,
        )
    return float(result.stdout.decode("utf-8").split("\n")[1].split()[-1])
