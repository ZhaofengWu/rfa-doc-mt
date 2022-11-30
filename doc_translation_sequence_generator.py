from typing import Dict, Optional

from fairseq.sequence_generator import SequenceGenerator
import torch
from torch import Tensor
import torch.nn as nn

from models.utils import subtensor_before


class DocTranslationSequenceGenerator(nn.Module):
    def __init__(
        self,
        sent_level_generator: SequenceGenerator,
        use_sep,
        tgt_dict,
        doc_bleu,
        duplicate_input_times,
    ):
        super().__init__()
        self.sent_level_generator = sent_level_generator
        self.eos = torch.LongTensor([self.sent_level_generator.eos])
        self.sep = torch.LongTensor([self.sent_level_generator.tgt_dict.sep()])
        args = self.sent_level_generator.model.models[0].args
        self.context_size = args.context_size
        self.full_supervision = args.full_supervision
        # TODO: can we replace the following with args.X?
        self.use_sep = use_sep
        self.tgt_dict = tgt_dict
        self.doc_bleu = doc_bleu
        self.duplicate_input_times = duplicate_input_times

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        assert prefix_tokens is None
        assert constraints is None

        pad_idx = sample["pad_idx"]
        src_tokens = sample["net_input"]["src_tokens"]
        src_tokens = [sent[sent != pad_idx] for sent in src_tokens]

        output = []
        context = []
        for i in range(len(src_tokens)):
            start_idx = max(i - self.context_size, 0)
            end_idx = min(i, len(src_tokens) - 1)
            curr_src_tokens = []
            for j in range(start_idx, end_idx + 1):
                tokens = src_tokens[j]
                if self.use_sep and j < end_idx:
                    tokens = torch.cat((tokens, self.sep.to(tokens)), dim=0)
                elif j == end_idx:
                    tokens = torch.cat((tokens, self.eos.to(tokens)), dim=0)
                curr_src_tokens.append(tokens)
            src_ctx_len = sum(len(tokens) for tokens in curr_src_tokens[: i - start_idx])
            curr_src_tokens = torch.cat(curr_src_tokens, dim=0).unsqueeze(0)

            prefix_tokens = None
            prefix_len = 0
            if len(context) > 0:
                prefix_tokens = torch.cat(context, dim=0).unsqueeze(0)
                prefix_len = prefix_tokens.shape[1]

            sent_sample = {
                "net_input": {
                    "src_tokens": curr_src_tokens,
                    "src_lengths": torch.LongTensor([curr_src_tokens.shape[1]]).to(curr_src_tokens),
                }
            }
            if self.duplicate_input_times > 1:
                sent_sample["net_input"]["src_tokens"] = sent_sample["net_input"]["src_tokens"].expand(
                    self.duplicate_input_times, -1
                )
                sent_sample["net_input"]["src_lengths"] = sent_sample["net_input"]["src_lengths"].expand(
                    self.duplicate_input_times, -1
                )
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens.expand(self.duplicate_input_times, -1)

            # We dynamically adjust max_len_b for each sentence to make sure that each sentence has
            # the correct a/b values
            orig_max_len_b = self.sent_level_generator.max_len_b
            self.sent_level_generator.max_len_b += (
                prefix_len - self.sent_level_generator.max_len_a * src_ctx_len
            )
            gen_out = self.sent_level_generator._generate(
                sent_sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )
            self.sent_level_generator.max_len_b = orig_max_len_b
            assert len(gen_out) == self.duplicate_input_times  # batch size
            tokens = gen_out[0][0]["tokens"]  # batch size, beam size
            assert tokens[-1].item() == self.eos
            assert len(tokens) > prefix_len
            # If the model outputs multiple sentences in one timestep, only keep the first. This
            # empirically leads to slightly better performance.
            tokens = subtensor_before(
                tokens[prefix_len:], (self.tgt_dict.eos(), self.tgt_dict.sep())
            )

            # Prepare context
            if self.context_size > 0:
                context.append(
                    torch.cat((tokens, self.sep.to(tokens)), dim=0) if self.use_sep else tokens
                )
                context = context[-self.context_size :]

            output.append(
                [
                    {
                        "tokens": torch.cat((tokens, self.eos.to(tokens)), dim=0),
                        "alignment": torch.FloatTensor([]),
                        "score": gen_out[0][0]["score"],
                        "positional_scores": gen_out[0][0]["positional_scores"][prefix_len:],
                    }
                ]
            )  # only output 1-best

        if self.doc_bleu:
            output = [
                [
                    {
                        "tokens": torch.cat(tuple(o[0]["tokens"] for o in output), dim=0),
                        "alignment": torch.FloatTensor([]),
                        "score": output[0][0]["score"],
                        "positional_scores": torch.cat(
                            tuple(o[0]["positional_scores"] for o in output), dim=0
                        ),
                    }
                ]
            ]
        return output
