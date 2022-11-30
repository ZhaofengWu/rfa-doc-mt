# From https://github.com/haopeng-uw/RFA/tree/6d59b1c79b47e78c9ece0ee4c3a97c3fb29488b1

""""Causal RFA."""

from typing import Dict, Optional

import torch
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter

from models.rfa import RFA
from models.utils import lens_to_mask
from utils.rfa_utils import upgrade_state_dict_named
from utils.rfa_utils import random_project
from utils.rfa_utils import build_random_matrices
from utils.rfa_utils import normalize_attn_weights
from utils.rfa_utils import EPS
from utils.rfa_utils import tau
import rfa_cuda



def cuda_incremental_rfa(*,
                    phi_q: Tensor,
                    phi_k: Tensor,
                    v: Tensor,
                    s: Tensor,
                    z: Tensor) -> Tensor:
    """Loop causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [tgt_len, bsz, num_heads, 2 * proj_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        s: [bsz, num_heads, 2 * proj_dim, head_dim]
        z: [bsz, num_heads, 2 * proj_dim]
    """
    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = v.size(-1)
    assert tgt_len == 1

    phi_q = phi_q.contiguous().view(tgt_len, bsz * num_heads, proj_dim)
    phi_k = phi_k.contiguous().view(tgt_len, bsz * num_heads, proj_dim)
    v = v.contiguous().view(tgt_len, bsz * num_heads, -1)
    s = s.contiguous().view(bsz * num_heads, proj_dim * head_dim)
    z = z.contiguous().view(bsz * num_heads, proj_dim)
    attn, s, z = rfa_cuda.causal_rfa(phi_q, phi_k, v, s, z)
    attn = attn.contiguous().view(tgt_len, bsz, num_heads * head_dim)
    s = s.contiguous().view(bsz, num_heads, proj_dim, head_dim)
    z = z.contiguous().view(bsz, num_heads, proj_dim)
    return attn, s, z


def incremental_rfa(*,
                    phi_q: Tensor,
                    phi_k: Tensor,
                    v: Tensor,
                    g0: Optional[Tensor] = None,
                    g1: Optional[Tensor] = None,
                    s: Optional[Tensor] = None,
                    z: Optional[Tensor] = None) -> Tensor:
    """Loop causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [tgt_len, bsz, num_heads, 2 * proj_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        g: [tgt_len, bsz, num_heads, 1]
        s: [bsz, num_heads, 2 * proj_dim, head_dim]
        z: [bsz, num_heads, 2 * proj_dim]
    """
    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = v.size(-1)

    if s is None:
        assert z is None
        s = torch.zeros(
            (bsz, num_heads, proj_dim, head_dim),
            device=v.device, dtype=v.dtype)
        z = torch.zeros((bsz, num_heads, proj_dim), device=v.device, dtype=v.dtype)
    has_g = g0 is not None
    if not has_g:
        assert g1 is None
    elif g1 is None:
        g1 = 1. - g0

    attns = []
    for i in range(tgt_len):
        s = (g0[i].unsqueeze(-1) if has_g else 1) * s + (
            g1[i].unsqueeze(-1) if has_g else 1
        ) * torch.einsum("bhk,bhd->bhkd", phi_k[i, ...], v[i, ...])
        z = (g0[i] if has_g else 1) * z + (g1[i] if has_g else 1) * phi_k[i, ...]
        qs = torch.einsum("bhk,bhkd->bhd", phi_q[i, ...], s)
        qz = torch.einsum("bhk,bhk->bh", phi_q[i, ...], z)
        qz = qz.clamp_min(EPS)

        # [bsz, num_heads, head_dim]
        attns.append(qs / qz.unsqueeze(-1))
    # [tgt_len, bsz, num_heads, head_dim]
    attns = torch.stack(attns, dim=0).contiguous().view(
        tgt_len, bsz, num_heads * head_dim)
    return attns, s, z


def masked_rfa(*,
               phi_q: Tensor,
               phi_k: Tensor,
               v: Tensor,
               key_padding_mask: Optional[Tensor] = None,
               attn_mask: Optional[Tensor] = None) -> Tensor:
    """Masked causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [tgt_len, bsz, num_heads, 2 * proj_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        attn_mask (ByteTensor, optional): typically used to
            implement causal attention, where the mask prevents the
            attention from looking forward in time (default: None).
            [tgt_len, src_len]
    Return:
        attn: [tgt_len, bsz, num_heads * head_dim]
    """

    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = v.size(-1)
    # This is part of a workaround to get around fork/join parallelism
    # not supporting Optional types.
    if key_padding_mask is not None and key_padding_mask.dim() == 0:
        key_padding_mask = None

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == tgt_len

    # [bsz, num_heads, tgt_len, src_len]
    attn_weights = torch.einsum("tbhk,sbhk->bhts", phi_q, phi_k)
    assert list(attn_weights.size()) == [bsz, num_heads, tgt_len, tgt_len]
    if key_padding_mask is not None:
        # [bsz, 1, 1, src_len]: bool
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
        attn_weights = attn_weights.masked_fill(mask, 0.0)

    if attn_mask is not None:
        # [tgt_len, src_len]: bool. Masked are True
        mask = (attn_mask < 0.0)
        # [1, 1, tgt_len, src_len]
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights.masked_fill(mask, 0.0)
    attn_weights = normalize_attn_weights(attn_weights, dtype=attn_weights.dtype)

    # [tgt_len, bsz, num_heads, head_dim]
    attn = torch.einsum("bhts,sbhd->tbhd", attn_weights, v)
    assert list(attn.size()) == [tgt_len, bsz, num_heads, head_dim]
    attn = attn.contiguous().view(tgt_len, bsz, num_heads * head_dim)
    return attn


def cuda_causal_rfa(*,
                    phi_q: Tensor,
                    phi_k: Tensor,
                    v: Tensor,
                    key_padding_mask: Optional[Tensor] = None
                    ) -> Tensor:
    """Cuda causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [tgt_len, bsz, num_heads, 2 * proj_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, tgt_len)`, where
                padding elements are indicated by 1s.
    Return:
        attn: [tgt_len, bsz, num_heads * head_dim]
    """

    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = v.size(-1)
    # This is part of a workaround to get around fork/join parallelism
    # not supporting Optional types.
    if key_padding_mask is not None and key_padding_mask.dim() == 0:
        key_padding_mask = None

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == tgt_len
        # [tgt_len, bsz]: bool
        mask = key_padding_mask.to(torch.bool).transpose(0, 1)
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # [tgt_len, bsz, 1, 1]
        phi_k = phi_k.masked_fill(mask, 0.0)

    phi_q = phi_q.contiguous().view(tgt_len, bsz * num_heads, -1)
    phi_k = phi_k.contiguous().view(tgt_len, bsz * num_heads, -1)
    v = v.contiguous().view(tgt_len, bsz * num_heads, head_dim)

    attn = RFA.apply(phi_q, phi_k, v)  # [tgt_len, bsz * num_heads, head_dim]
    attn = attn.contiguous().view(tgt_len, bsz, num_heads * head_dim)
    return attn


@with_incremental_state
class CausalAttention(nn.Module):
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
        self.proj_dim = args.causal_proj_dim
        self.reparam_proj = reparam_proj
        self.learned_tau = args.learned_tau
        self.norm_rescale = args.norm_rescale
        self.cuda_causal_rfa = args.cuda_causal_rfa
        self.use_sentential_gate = args.use_sentential_gate
        self.decay_sentential_gate = args.decay_sentential_gate
        self.decay_gate_bias = args.decay_gate_bias
        self.cuda_inference = args.cuda_inference
        assert not (self.learned_tau and self.reparam_proj)
        assert not (self.use_sentential_gate and self.decay_sentential_gate)
        if self.use_sentential_gate or self.decay_sentential_gate:
            assert args.use_sep

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        bias = True
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.k_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=True), q_noise, qn_block_size
        )
        if self.use_sentential_gate or self.decay_sentential_gate:
            self.g_proj = quant_noise(
                nn.Linear(embed_dim, num_heads, bias=True), q_noise, qn_block_size
            )
        if reparam_proj:
            self.sigma = Parameter(Tensor(num_heads, 1, head_dim))
        if self.learned_tau:
            self.tau = Parameter(Tensor(self.num_heads, 1, 1))
        else:
            self.tau = tau

        self.reset_parameters(args)
        self.upgrade_state_dict_named = upgrade_state_dict_named

    def reset_parameters(self, args):
        gain = args.init_scale ** -0.5
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        if self.use_sentential_gate or self.decay_sentential_gate:
            nn.init.xavier_uniform_(self.g_proj.weight, gain=gain)
            nn.init.constant_(
                self.g_proj.bias, self.decay_gate_bias if self.decay_sentential_gate else 0.0
            )

        # std = 0.02 * args.init_scale ** -0.5
        # nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        # nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        # nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        # nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)

        if self.reparam_proj:
            nn.init.constant_(self.sigma, 1.)
        if self.learned_tau:
            nn.init.constant_(self.tau, 1.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        # if self.q_proj.bias is not None:
        #     nn.init.constant_(self.q_proj.bias, 0.0)
        #     nn.init.constant_(self.k_proj.bias, 0.0)
        #     nn.init.constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        random_matrices: Tensor,
        sep_indices: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights=False,  # unused
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            x: [tgt_len, bsz, embed_dim]
            random_matrices: [num_heads, proj_dim, head_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
                [tgt_len, src_len]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        assert list(x.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        g0: Optional[Tensor] = None
        g1: Optional[Tensor] = None

        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        k = k.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        v = v.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        random_matrices = build_random_matrices(
            random_matrices=random_matrices,
            tau=tau(self.tau),
            sigma=self.sigma if self.reparam_proj else None,
            reparam_proj=self.reparam_proj)
        phi_q, phi_k = random_project(
            random_matrices=random_matrices,
            x=q,
            y=k,
            use_cuda=self.cuda_inference,
            random_feature=self.random_feature)
        # phi_q = random_project(
        #     x=q,
        #     random_matrices=random_matrices,
        #     random_feature=self.random_feature,
        #     random_feature=self.random_feature
        # )
        # phi_k = random_project(
        #     x=k,
        #     random_matrices=random_matrices,
        #     random_feature=self.random_feature
        # )

        if saved_state is not None:
            # Incremental decoding
            assert tgt_len == 1
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_s" in saved_state:
                assert "prev_z" in saved_state
                prev_s = saved_state["prev_s"]
                prev_z = saved_state["prev_z"]
                assert prev_s is not None
                assert prev_z is not None
                if self.use_sentential_gate:
                    prev_g = saved_state.get("prev_g")
                    assert prev_g is not None
            else:
                proj_dim = phi_q.size(-1)
                prev_s = x.new_zeros(
                    bsz, self.num_heads, proj_dim, self.head_dim)
                prev_z = x.new_zeros(
                    bsz, self.num_heads, proj_dim)
                if self.use_sentential_gate:
                    prev_g = x.new_zeros(bsz, self.num_heads)

            if self.use_sentential_gate or self.decay_sentential_gate:
                # sep_indices: (bsz, max_seps)
                max_seps = sep_indices.shape[1]
                assert max_seps in {0, 1}
                assert ((sep_indices == 0) | (sep_indices == -1)).all()
                if max_seps == 0:
                    is_sep = x.new_zeros(bsz, dtype=torch.bool)
                else:
                    is_sep = sep_indices.squeeze(1) == 0
                g = self.g_proj(x.squeeze(0))  # [bsz, num_heads]
                g = torch.sigmoid(g).where(is_sep.unsqueeze(-1), g.new_ones(()))
                g0 = torch.ones_like(g).unsqueeze(0).unsqueeze(-1)
                g1 = torch.ones_like(g).unsqueeze(0).unsqueeze(-1)
                if not self.decay_sentential_gate:
                    g1 = (1 - prev_g).unsqueeze(0).unsqueeze(-1)
                    saved_state["prev_g"] = g.where(is_sep.unsqueeze(-1), prev_g)

            # [tgt_len, bsz, embed_dim]
            if self.cuda_inference:
                attn, s, z = cuda_incremental_rfa(phi_q=phi_q, phi_k=phi_k, v=v, s=prev_s, z=prev_z)
            else:
                attn, s, z = incremental_rfa(
                    phi_q=phi_q, phi_k=phi_k, v=v, g0=g0, g1=g1, s=prev_s, z=prev_z)

            if self.use_sentential_gate or self.decay_sentential_gate:
                s = s * g.unsqueeze(-1).unsqueeze(-1)
                z = z * g.unsqueeze(-1)
            saved_state["prev_s"], saved_state["prev_z"] = s, z
            # In this branch incremental_state is never None
            # assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        else:
            def attn_fn(phi_q, phi_k, v):
                kwargs = dict(phi_q=phi_q, phi_k=phi_k, v=v, key_padding_mask=key_padding_mask)
                if not self.cuda_causal_rfa:
                    kwargs['attn_mask'] = attn_mask
                # [tgt_len, bsz, embed_dim]
                return (cuda_causal_rfa if self.cuda_causal_rfa else masked_rfa)(**kwargs)

            if self.use_sentential_gate or self.decay_sentential_gate:
                # sep_indices: (bsz, max_seps)
                max_seps = sep_indices.shape[1]
                if max_seps > 1:
                    assert (
                        (sep_indices[:, 1:] - sep_indices[:, :-1] > 0)
                        | (sep_indices[:, 1:] == -1)
                    ).all()  # sorted or padding

                context_mask = x.new_zeros(tgt_len, bsz, 1, 1, dtype=torch.bool)
                gated_context_mask = x.new_zeros(tgt_len, bsz, self.num_heads, 1)
                prev_g = x.new_zeros(1, bsz, self.num_heads, 1)
                attn = x.new_zeros(tgt_len, bsz, embed_dim)
                for sep_idx in range(max_seps):
                    curr_sep_indices = sep_indices[:, sep_idx]
                    cum_mask = lens_to_mask(
                        curr_sep_indices + 1, max_len=tgt_len
                    ).transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [tgt_len, bsz, 1, 1]
                    cum_mask = cum_mask | context_mask  # for padding where cum_mask has len 0
                    curr_mask = cum_mask & ~context_mask
                    gated_cum_mask = gated_context_mask * prev_g + curr_mask.to(
                        gated_context_mask
                    ) * (1 if self.decay_sentential_gate else (1 - prev_g))
                    attn = attn + attn_fn(
                        phi_q * curr_mask, phi_k * gated_cum_mask, v * cum_mask
                    )

                    sep_emb = x[curr_sep_indices.clamp(min=0), torch.arange(bsz)]
                    g = self.g_proj(sep_emb).unsqueeze(0).unsqueeze(-1)
                    g = torch.sigmoid(g)
                    prev_g = g.where(curr_sep_indices.reshape(1, -1, 1, 1) >= 0, prev_g)
                    gated_context_mask = gated_cum_mask
                    context_mask = cum_mask

                # last sentence
                if key_padding_mask is not None:
                    cum_mask = ~key_padding_mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
                else:
                    # There's no padding in this case. I think it's a stupid design on fairseq's
                    # part to not use this mask when there's no padding, but here we are
                    cum_mask = x.new_ones(tgt_len, bsz, 1, 1, dtype=torch.bool)
                curr_mask = cum_mask & ~context_mask
                gated_cum_mask = gated_context_mask * prev_g + curr_mask.to(
                    gated_context_mask
                ) * (1 if self.decay_sentential_gate else (1 - prev_g))
                attn = attn + attn_fn(
                    phi_q * curr_mask, phi_k * gated_cum_mask, v * cum_mask
                )
            else:
                # [tgt_len, bsz, embed_dim]
                attn = attn_fn(phi_q, phi_k, v)

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
