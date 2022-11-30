# From https://github.com/haopeng-uw/RFA/tree/6d59b1c79b47e78c9ece0ee4c3a97c3fb29488b1

""""Cross RFA."""

from typing import Dict, List, Optional, Tuple

import torch
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter

from models.rfa_transformer_encoder import RFACrossAttentionState
from utils.rfa_utils import upgrade_state_dict_named
from utils.rfa_utils import random_project
from utils.rfa_utils import load_random_matrices
from utils.rfa_utils import sample_random_matrices
from utils.rfa_utils import build_random_matrices
from utils.rfa_utils import EPS
from utils.rfa_utils import tau
import rfa_cuda


class CrossAttentionProjectLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.random_feature = args.random_feature
        self.embed_dim = args.decoder_embed_dim
        self.num_heads = args.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.proj_dim = args.cross_proj_dim
        self.reparam_proj = args.reparam_proj
        self.learned_tau = args.learned_tau
        self.norm_rescale = args.norm_rescale
        assert not (self.learned_tau and self.reparam_proj)
        self.bias = True
        q_noise = args.quant_noise_pq
        qn_block_size = args.quant_noise_pq_block_size
        self.k_proj = quant_noise(
            nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias), q_noise, qn_block_size
        )
        if self.reparam_proj:
            self.sigma = Parameter(Tensor(self.num_heads, 1, self.head_dim))
        if self.learned_tau:
            self.tau = Parameter(Tensor(self.num_heads, 1, 1))
        else:
            self.tau = args.cross_tau
        self.decay_gate_bias = args.decay_gate_bias
        self.cuda_inference = args.cuda_inference

        self.reset_parameters(args)
        self.upgrade_state_dict_named = upgrade_state_dict_named

    def reset_parameters(self, args):
        gain = args.init_scale ** -0.5
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        # std = 0.02 * args.init_scale ** -0.5
        # nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        # nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        if self.reparam_proj:
            nn.init.constant_(self.sigma, 1.)
        if self.learned_tau:
            nn.init.constant_(self.tau, 1.)
        # if self.k_proj.bias is not None:
        #     nn.init.constant_(self.k_proj.bias, 0.0)
        #     nn.init.constant_(self.v_proj.bias, 0.0)

    def project_and_reshape(
        self,
        *,
        encoder_output: Tensor,
        random_matrices: Tensor,
        mask: Optional[Tensor] = None,
        sep_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            encoder_output: [seq_len, bsz, embed_dim]
            random_matrices: num_layers * [num_heads, proj_dim, head_dim]
            mask: [src_len, bsz, 1, 1]: bool
        Return:
            s: [bsz, num_heads, proj_dim, head_dim]
            z: [bsz, num_heads, proj_dim]
        Einsum notations:
            b: bsz
            s: seq_len
            n: num_layers
            h: num_heads
            k: proj_dim
            d: head_dim
        """
        src_len, bsz, _ = encoder_output.size()

        # [src_len, bsz, num_head * head_dim]
        projected_k = self.k_proj(encoder_output)
        projected_v = self.v_proj(encoder_output)

        # [src_len, bsz, num_heads, head_dim]
        projected_k = projected_k.contiguous().view(
            src_len, bsz, self.num_heads, self.head_dim)
        projected_v = projected_v.contiguous().view(
            src_len, bsz, self.num_heads, self.head_dim)
        random_matrices = build_random_matrices(
            random_matrices=random_matrices,
            tau=tau(self.tau),
            sigma=self.sigma if self.reparam_proj else None,
            reparam_proj=self.reparam_proj)

        # [seq_len, bsz, num_heads, 2 * proj_dim]
        phi_k = random_project(
            x=projected_k,
            random_matrices=random_matrices,
            norm_rescale=self.norm_rescale,
            tau=tau(self.tau),
            scale=self.head_dim ** -0.5,
            random_feature=self.random_feature
        )
        if mask is not None:
            # mask: [src_len, bsz, 1, 1]
            # phi_k: [src_len, bsz, num_heads, 2 * proj_dim]
            phi_k = phi_k.masked_fill(mask, 0.0)
        if not self.cuda_inference:
            # [bsz, num_heads, proj_dim, head_dim]
            s = torch.einsum("sbhk,sbhd->bhkd", phi_k, projected_v)
            z = torch.sum(phi_k, dim=0)  # [bsz, num_heads, head_dim]
        else:
            assert not self.training
            phi_k = phi_k.contiguous().view(src_len, bsz * self.num_heads, -1)
            projected_v = projected_v.contiguous().view(src_len, bsz * self.num_heads, -1)
            s, z = rfa_cuda.calculate_sz(phi_k, projected_v)
            s = s.contiguous().view(bsz, self.num_heads, self.head_dim, -1)
            z = z.contiguous().view(bsz, self.num_heads, -1)
        return s, z, random_matrices

    def forward(
        self,
        *,
        encoder_output: Tensor,
        random_matrices: Tensor,
        mask: Optional[Tensor] = None,
        sep_indices: Optional[Tensor] = None,
    ) -> RFACrossAttentionState:
        """
        Args:
            encoder_output: [seq_len, bsz, embed_dim]
            random_matrices: [num_heads, proj_dim, head_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        seq_len, bsz, embed_dim = encoder_output.size()
        assert embed_dim == self.embed_dim

        s, z, random_matrices = self.project_and_reshape(
            encoder_output=encoder_output,
            random_matrices=random_matrices,
            mask=mask,
            sep_indices=sep_indices,
        )
        return RFACrossAttentionState(s, z, random_matrices)


class CrossAttentionProject(nn.Module):
    """Encoder output projection for random feature cross attention."""
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.decoder_layers
        self.embed_dim = args.decoder_embed_dim
        self.num_heads = args.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.proj_dim = args.cross_proj_dim
        assert self.num_heads * self.head_dim == self.embed_dim

        self.layers = [CrossAttentionProjectLayer(args) for _ in range(self.num_layers)]
        self.layers = nn.ModuleList(self.layers)

        self.random_matrices = load_random_matrices(
            random_matrices_path=args.random_matrices_path,
            head_dim=self.head_dim,
            proj_dim=self.proj_dim,
            dtype=torch.float16 if args.fp16 else torch.float32)
        self.random_matrices_eval = sample_random_matrices(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            random_matrices=self.random_matrices,
            is_training=False)

        self.random_matrices_eval = nn.Parameter(self.random_matrices_eval)

    def forward(
        self,
        *,
        encoder_output: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        sep_indices: Optional[Tensor] = None,
    ) -> List[RFACrossAttentionState]:
        """
        Args:
            encoder_output: [seq_len, bsz, embed_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        src_len, bsz, embed_dim = encoder_output.size()
        assert embed_dim == self.embed_dim

        random_matrices = sample_random_matrices(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            random_matrices=self.random_matrices,
            is_training=True) if self.training else self.random_matrices_eval
        mask = key_padding_mask
        if mask is not None and mask.dim() == 0:
            mask = None
        if mask is not None:
            mask = mask.transpose(0, 1)
            assert mask.size(0) == src_len and mask.size(1) == bsz
            # [src_len, bsz, 1, 1]: bool
            mask = mask.unsqueeze(-1).unsqueeze(-1)
        states = []
        for i in range(self.num_layers):
            states.append(self.layers[i](
                encoder_output=encoder_output,
                random_matrices=random_matrices[i] if self.training else random_matrices,
                mask=mask,
                sep_indices=sep_indices))
        return states


@with_incremental_state
class CrossAttention(nn.Module):
    """Random feature cross attention."""

    def __init__(
        self,
        *,
        args,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        tau: float = 1.0,
        reparam_proj: bool = False
    ):
        super().__init__()
        self.random_feature = args.random_feature
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bias = True
        self.cuda_inference = args.cuda_inference

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=self.bias), q_noise, qn_block_size
        )
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=True), q_noise, qn_block_size
        )

        self.reset_parameters(args)
        self.upgrade_state_dict_named = upgrade_state_dict_named

    def reset_parameters(self, args):
        # gain = 1. / math.sqrt(args.init_scale)
        # nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        # nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)

        std = 0.02 * args.init_scale ** -0.5
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        state: RFACrossAttentionState,
        sep_indices: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            state: s, z, random_matrices
                s [bsz, num_heads, 2 * proj_dim, head_dim]
                z [bsz, num_heads, 2 * proj_dim]
                random_matrices: [num_heads, proj_dim, head_dim]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        s, z, random_matrices = state
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        # [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_q = random_project(
            random_matrices=random_matrices,
            x=q,
            use_cuda=self.cuda_inference,
            random_feature=self.random_feature
        )

        def attn_fn(phi_q_, s_, z_):
            if self.cuda_inference:
                assert not self.training and phi_q_.size(0) == 1  # decoding
                phi_q_ = phi_q_.contiguous().view(tgt_len, bsz * self.num_heads, -1)
                s_ = s_.contiguous().view(bsz * self.num_heads, self.head_dim, -1)
                z_ = z_.contiguous().view(bsz * self.num_heads, -1)
                attn = rfa_cuda.cross_rfa(phi_q_, s_, z_)
                return attn.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim)
            else:
                qs = torch.einsum("tbhk,bhkd->tbhd", phi_q_, s_)
                qz = torch.einsum("tbhk,bhk->tbh", phi_q_, z_).clamp_min(EPS)
                # [tgt_len, bsz, num_heads, head_dim]
                return qs / qz.unsqueeze(-1)

        attn = attn_fn(phi_q, s, z)

        assert list(attn.size()) == [tgt_len, bsz, self.num_heads, self.head_dim]
        attn = attn.contiguous().view(tgt_len, bsz, self.num_heads * self.head_dim)
        # [tgt_len, bsz, embed_dim]
        attn = self.out_proj(attn)
        return attn

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
