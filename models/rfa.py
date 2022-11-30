"""RFA cuda.

Einsum notations:
    b: bsz
    s: seq_len
    n: num_layers
    h: num_heads
    k: proj_dim
    d: head_dim
"""

from typing import  Tuple
from rfa_cuda import random_project as random_project_cuda
from rfa_cuda import random_project_xy as random_project_xy_cuda
import torch
from torch import Tensor
import rfa_cuda


EPS = 1.0


def reverse_cumsum(x, dim):
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])


def rfa_debug(q, k, v):
    """
    Args:
        q: [tgt_len, bsz * num_heads, proj_dim]
        k: [tgt_len, bsz * num_heads, proj_dim]
        v: [tgt_len, bsz * num_heads, head_dim]

    Return:
        attn: [tgt_len, bsz * num_heads, head_dim]
    """
    s = torch.einsum("tbk,tbd->tbkd", k, v)
    s = torch.cumsum(s, dim=0)
    qs = torch.einsum("tbkd,tbk->tbd", s, q)

    z = torch.cumsum(k, dim=0)
    qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)
    attn = qs / qz.unsqueeze(-1)
    return attn



def _normalize(x: Tensor) -> Tuple[Tensor, Tensor]:
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return torch.div(x, norm + 1e-3)
    
    
def rff(x: Tensor, w: Tensor) -> Tensor:
    tgt_len, bsz, num_heads, head_dim = x.shape
    proj_dim = w.size(1)
    x = _normalize(x)
    x = torch.einsum("hkd,tbhd->tbhk", w, x)
    x = x.contiguous().view(tgt_len, bsz * num_heads, proj_dim)
    sin_x, cos_x = torch.sin(x), torch.cos(x)
    phi_x = torch.cat([sin_x.unsqueeze(-1), cos_x.unsqueeze(-1)], dim=-1)
    phi_x = phi_x.contiguous().view(tgt_len, bsz * num_heads, 2 * proj_dim)
    return phi_x
    
    
def random_project(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    cuda: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        q: [1, bsz, num_heads, head_dim]
        k: [1, bsz, num_heads, head_dim]
        w: [num_heads, proj_dim, head_dim]
        b: [num_heads, proj_dim]
    Return:
        phi_q: [1, bsz, num_heads, proj_dim]
        phi_k: [1, bsz, num_heads, proj_dim]
    """
    if cuda:
        # phi_q = random_project_cuda(q, w)
        # phi_k = random_project_cuda(k, w)
        phi_q, phi_k = random_project_xy_cuda(q, k, w)
    else:
        phi_q, phi_k = rff(q, w), rff(k, w)
    return phi_q, phi_k


def causal_rfa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    s: Tensor,
    z: Tensor,
    cuda: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        q: [1, bsz * num_heads, proj_dim]
        k: [1, bsz * num_heads, proj_dim]
        v: [1, bsz * num_heads, head_dim]
        s: [bsz * num_heads, proj_dim, head_dim]
        z: [bsz * num_heads, proj_dim]
    Return:
        attn: [tgt_len, bsz * num_heads, head_dim]
        s: [bsz * num_heads, head_dim, proj_dim]
        z: [bsz * num_heads, proj_dim]
    """
    if cuda:
        attn, s, z = rfa_cuda.causal_rfa(q, k, v, s, z)
    else:
        k = k.squeeze(0)
        v = v.squeeze(0)
        # [bsz * num_heads, proj_dim, head_dim]
        s = s + torch.einsum("bk,bd->bkd", k, v)
        z = z + k

        qs = torch.einsum("bkd,tbk->tbd", s, q)
        qz = torch.einsum("tbk,bk->tb", q, z).clamp_min(EPS)
        attn = qs / qz.unsqueeze(-1)
    return attn, s, z


def calculate_sz(
    k: Tensor,
    v: Tensor,
    cuda: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        k: [src_len, bsz * num_heads, proj_dim]
        v: [src_len, bsz * num_heads, head_dim]
    Return:
        s: [bsz * num_heads, head_dim, proj_dim]
        z: [bsz * num_heads, proj_dim]
    """
    if cuda:
        s, z = rfa_cuda.calculate_sz(k, v)
    else:
        s = torch.einsum("tbk,tbd->bkd", k, v)
        z = torch.sum(k, dim=0)
    return s, z


def cross_rfa(
    q: Tensor,
    s: Tensor,
    z: Tensor,
    cuda: bool = False,
) -> Tensor:
    """
    Args:
        q: [tgt_len, bsz * num_heads, proj_dim]
        s: [bsz * num_heads, head_dim, proj_dim]
        z: [bsz * num_heads, proj_dim]
    Return:
        attn: [tgt_len, bsz * num_heads, head_dim]
    """
    if cuda:
        attn = rfa_cuda.cross_rfa(q, s, z)
    else:
        qs = torch.einsum("bkd,tbk->tbd", s, q)
        qz = torch.einsum("tbk,bk->tb", q, z).clamp_min(EPS)
        attn = qs / qz.unsqueeze(-1)
    return attn


def softmax_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor
) -> Tensor:
    """
    Args:
        q: [1, bsz * num_heads, proj_dim]
        k: [tgt_len, bsz * num_heads, proj_dim]
        v: [tgt_len, bsz * num_heads, head_dim]
        s: [bsz * num_heads, proj_dim, head_dim]
        z: [bsz * num_heads, proj_dim]
    Return:
        s: [bsz * num_heads, proj_dim, head_dim]
        z: [bsz * num_heads, proj_dim]
    """
    attn_weights = torch.einsum("tbk,sbk->bts", q, k)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn = torch.einsum("bts,sbd->tbd", attn_weights, v)
    return attn


class RFA(torch.autograd.Function):

    @staticmethod
    def forward_torch(q, k, v):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]

        Return:
            attn: [tgt_len, bsz * num_heads, head_dim]
        """
        s = torch.einsum("tbk,tbd->tbkd", k, v)
        s = torch.cumsum(s, dim=0)
        qs = torch.einsum("tbkd,tbk->tbd", s, q)

        z = torch.cumsum(k, dim=0)
        qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)
        attn = qs / qz.unsqueeze(-1)
        return attn

    @staticmethod
    def backward_torch(q, k, v, grad_attn):
        """
        Args:
            grad_attn: [tgt_len, bsz * num_heads, head_dim]
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]
        Return:
            grad_q: [tgt_len, bsz * num_heads, proj_dim]
            grad_k: [tgt_len, bsz * num_heads, proj_dim]
            grad_v: [tgt_len, bsz * num_heads, head_dim]
        """
        s = torch.einsum("tbk,tbd->tbkd", k, v)
        s = torch.cumsum(s, dim=0)
        qs = torch.einsum("tbkd,tbk->tbd", s, q)

        z = torch.cumsum(k, dim=0)
        qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)

        # [bsz, tgt_len, head_dim]
        grad_qs = grad_attn / qz.unsqueeze(-1)

        grad_qz = torch.einsum("tbd,tbd->tb", grad_attn, qs)
        grad_qz = -grad_qz / (qz ** 2)
        grad_qz = grad_qz * (qz > EPS)

        grad_q = torch.einsum("tbd,tbkd->tbk", grad_qs, s) \
            + grad_qz.unsqueeze(-1) * z

        grad_s = torch.einsum("tbk,tbd->tbkd", q, grad_qs)
        grad_s = reverse_cumsum(grad_s, dim=0)
        grad_k = torch.einsum("tbkd,tbd->tbk", grad_s, v)
        grad_v = torch.einsum("tbkd,tbk->tbd", grad_s, k)

        grad_k = grad_k + reverse_cumsum(q * grad_qz.unsqueeze(-1), dim=0)

        return grad_q, grad_k, grad_v

    @staticmethod
    def forward_cuda(q, k, v):
        return rfa_cuda.forward(q, k, v)

    @staticmethod
    def backward_cuda(q, k, v, grad_attn):
        return rfa_cuda.backward(q, k, v, grad_attn)

    @staticmethod
    def forward(ctx, q, k, v):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]

        Return:
            attn: [tgt_len, bsz * num_heads, head_dim]
        """
        ctx.save_for_backward(q, k, v)
        attn = RFA.forward_cuda(q, k, v)
        # attn = RFA.forward_torch(q, k, v)
        return attn

    @staticmethod
    def backward(ctx, grad_attn):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]
            grad_attn: [tgt_len, bsz * num_heads, head_dim]
        Return:
            grad_q: [tgt_len, bsz * num_heads, proj_dim]
            grad_k: [tgt_len, bsz * num_heads, proj_dim]
            grad_v: [tgt_len, bsz * num_heads, head_dim]
        """
        q, k, v = ctx.saved_tensors
        grad_q, grad_k, grad_v = RFA.backward_cuda(q, k, v, grad_attn)
        # grad_q, grad_k, grad_v = RFA.backward_torch(q, k, v, grad_attn)
        return grad_q, grad_k, grad_v


if __name__ == "__main__":
    device = torch.device("cuda:0")
    dtype = torch.double

    bsz, tgt_len, proj_dim, head_dim = 2, 15, 128, 8
    q = torch.rand(
        (tgt_len, bsz, head_dim),
        device=device, dtype=dtype, requires_grad=True) - 0.5
    k = torch.rand(
        (tgt_len, bsz, head_dim),
        device=device, dtype=dtype, requires_grad=True) - 0.5
    v = torch.rand(
        (tgt_len, bsz, head_dim),
        device=device, dtype=dtype, requires_grad=True)

    res = torch.autograd.gradcheck(
        RFA.apply,
        (q, k, v),
        raise_exception=True)
