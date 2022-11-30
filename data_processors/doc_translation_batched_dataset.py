from data_processors.doc_translation_dataset import DocTranslationDataset
from fairseq_adaptations import collate


class DocTranslationBatchedDataset(DocTranslationDataset):
    def __getitem__(self, index):
        return super().__getitem__(index, strip_eos=False)

    def collater(self, samples, pad_to_length=None):
        new_samples = []
        for sample in samples:
            assert len(sample["source"]) == len(sample["target"]) == 4
            for i in range(4):
                new_samples.append(
                    {
                        "id": sample["id"] * 4 + i,
                        "source": sample["source"][i],
                        "target": sample["target"][i],
                    }
                )

        return collate(
            new_samples,
            pad_idx=self.sent_level_dataset.src_dict.pad(),
            eos_idx=self.sent_level_dataset.eos,
            left_pad_source=self.sent_level_dataset.left_pad_source,
            left_pad_target=self.sent_level_dataset.left_pad_target,
            input_feeding=self.sent_level_dataset.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.sent_level_dataset.pad_to_multiple,
        )

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        from fairseq.data import data_utils

        return data_utils.batch_by_size(
            indices,
            num_tokens_fn=lambda i: sum(
                self.sent_level_dataset.num_tokens(j) for j in self.doc_indices[i]
            ),
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=None,
        )
