# From https://github.com/haopeng-uw/RFA/tree/6d59b1c79b47e78c9ece0ee4c3a97c3fb29488b1

"""RFA Utils."""
from typing import Optional, Text, Tuple
import numpy as np
import torch
from torch import Tensor
from rfa_cuda import random_project as random_project_cuda
from rfa_cuda import random_project_xy as random_project_xy_cuda


EPS = 0.1
SCALE = 0.1

def load_random_matrices(
        *,
        random_matrices_path: str,
        head_dim: int,
        proj_dim: int,
        dtype: torch.dtype = torch.half) -> Tensor:
    # [num_random_matrices, proj_dim, head_dim]
    random_matrices = np.load(
        f"{random_matrices_path}/{head_dim}_{proj_dim}.npy")
    return torch.nn.Parameter(
        torch.tensor(random_matrices, dtype=dtype), requires_grad=False)


def sample_random_matrices(
        *,
        num_layers: int,
        num_heads: int,
        random_matrices: Tensor,
        is_training: bool = True):
    # random_matrices
    # [num_random_matrices, proj_dim, head_dim]

    if is_training:
        num_random_matrices = random_matrices.size(0)
        indices = np.random.choice(
            num_random_matrices,
            size=num_layers * num_heads,
            replace=False)
        # [num_layers * num_heads, proj_dim, head_dim]
        random_matrices = random_matrices[indices]
        sampled_random_matrices = []
        for i in range(num_layers):
            sampled_random_matrices.append(
                random_matrices[i * num_heads: (i + 1) * num_heads])
        return sampled_random_matrices
    else:
        indices = list(range(num_heads))
        # [num_layers * num_heads, proj_dim, head_dim]
        return random_matrices[indices]


def build_random_matrices(
        random_matrices: Tensor,
        tau: Tensor,
        sigma: Optional[Tensor] = None,
        reparam_proj: bool = False) -> Tensor:
    if reparam_proj:
        random_matrices = sigma * random_matrices
    return torch.div(random_matrices, tau)


def _normalize(x: Tensor) -> Tuple[Tensor, Tensor]:
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return torch.div(x, norm + 1e-3), norm


def tau(x: Tensor) -> Tensor:
    return x
    # tau \in (0.5, 1.5) for better numerical stability
    # return torch.sigmoid(x) + 0.5


def random_project(
        *,
        random_matrices: Tensor,
        x: Tensor,
        y: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        norm_rescale: bool = False,
        scale: float = 1.0,
        use_cuda: bool = False,
        random_feature: Text = "rrf"):
    # x: [seq_len, bsz, num_heads, head_dim]
    # random_matrices: [num_heads, proj_dim, head_dim]
    if random_feature == "rrf":
        def rrf(x: Tensor) -> Tensor:
            length, bsz, num_heads, _ = x.size()
            x, x_norm = _normalize(x)
            # [seq_len, bsz, num_heads, proj_dim]
            x = torch.einsum("sbhd,hkd->sbhk", x, random_matrices)
            x_sin, x_cos = torch.sin(x), torch.cos(x)
            phi_x = torch.cat([x_sin.unsqueeze(-1), x_cos.unsqueeze(-1)], dim=-1)
            phi_x = phi_x.contiguous().view(length, bsz, num_heads, -1) * SCALE
            if norm_rescale:
                nonlocal tau
                if type(tau) is Tensor or type(tau) is torch.nn.Parameter:
                    # tau: [num_heads, 1, 1]
                    # x_norm: [seq_len, bsz, num_heads, 1]
                    tau = tau.contiguous().view(1, 1, tau.size(0), 1)

                x_norm = 0.5 * (x_norm * scale / tau) ** 2.
                maxes = torch.max(x_norm, dim=0, keepdim=True).values.detach()
                phi_x = phi_x * torch.exp(x_norm - maxes)
            return phi_x
        if use_cuda:
            assert x.size(0) == 1 and tau is None and not norm_rescale and scale == 1.0
            if y is not None:
                x, y = random_project_xy_cuda(x, y, random_matrices)
                return x * SCALE, y * SCALE
            else:
                return random_project_cuda(x, random_matrices) * SCALE
        else:
            return (rrf(x), rrf(y)) if y is not None \
            else rrf(x)
    elif random_feature == "prf":
        scale = x.size(-1) ** -0.25
        x = x * scale
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True) ** 2

        # [seq_len, bsz, num_heads, proj_dim]
        x = torch.einsum("sbhd,hkd->sbhk", x, random_matrices)
        maxes = torch.max(x, dim=-1, keepdim=True).values
        x = x - 0.5 * x_norm - maxes
        phi_x = torch.exp(x) * SCALE
        return phi_x
    else:
        assert False, "random project setting can either be `rrf` or `ppf`"


def normalize_attn_weights(
        x: Tensor,
        dim: int = -1,
        dtype: torch.dtype = torch.float32) -> Tensor:
    x = x.type(torch.float32)
    # [..., 1]
    s = x.sum(dim=dim, keepdim=True).clamp(EPS)
    return torch.div(x, s).type(dtype)


def append_prev_key_padding_mask(
    key_padding_mask: Optional[Tensor],
    prev_key_padding_mask: Optional[Tensor],
    batch_size: int,
    src_len: int,
    static_kv: bool,
) -> Optional[Tensor]:
    # saved key padding masks have shape (bsz, seq_len)
    if prev_key_padding_mask is not None and static_kv:
        new_key_padding_mask = prev_key_padding_mask
    elif prev_key_padding_mask is not None and key_padding_mask is not None:
        new_key_padding_mask = torch.cat(
            [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
        )
    # During incremental decoding, as the padding token enters and
    # leaves the frame, there will be a time when prev or current
    # is None
    elif prev_key_padding_mask is not None:
        filler = torch.zeros(
            (batch_size, src_len - prev_key_padding_mask.size(1)),
            device=prev_key_padding_mask.device,
        )
        new_key_padding_mask = torch.cat(
            [prev_key_padding_mask.float(), filler.float()], dim=1
        )
    elif key_padding_mask is not None:
        filler = torch.zeros(
            (batch_size, src_len - key_padding_mask.size(1)),
            device=key_padding_mask.device,
        )
        new_key_padding_mask = torch.cat(
            [filler.float(), key_padding_mask.float()], dim=1
        )
    else:
        new_key_padding_mask = prev_key_padding_mask
    return new_key_padding_mask


def upgrade_state_dict_named(state_dict, name):
    prefix = name + "." if name != "" else ""
    items_to_add = {}
    keys_to_remove = []
    for k in state_dict.keys():
        if k.endswith(prefix + "in_proj_weight"):
            # in_proj_weight used to be q + k + v with same dimensions
            dim = int(state_dict[k].shape[0] / 3)
            items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
            items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
            items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

            keys_to_remove.append(k)

            k_bias = prefix + "in_proj_bias"
            if k_bias in state_dict.keys():
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                    dim: 2 * dim
                ]
                items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                keys_to_remove.append(prefix + "in_proj_bias")

    for k in keys_to_remove:
        del state_dict[k]

    for key, value in items_to_add.items():
        state_dict[key] = value
