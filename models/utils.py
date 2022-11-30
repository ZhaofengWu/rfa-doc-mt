import torch


def merge_first_dims(tensor, n=2):
    return tensor.reshape(-1, *tensor.shape[n:])


def split_first_dim(tensor, dims):
    return tensor.reshape(*dims, *tensor.shape[1:])


def unmask(tensor, mask):
    """
    unmask(t[mask], mask) == t
    """
    new_tensor = torch.zeros(*mask.shape, *tensor.shape[1:]).to(tensor)
    new_tensor[mask] = tensor
    return new_tensor


def subtensor_before(tensor, symbols):
    assert tensor.dim() == 1
    mask = tensor == symbols[0]
    for symbol in symbols[1:]:
        mask = mask | (tensor == symbol)
    nonzero = mask.nonzero()
    assert len(nonzero) > 0
    return tensor[: nonzero[0][0]]


def padded_nonzero(tensor):
    """
    padded_nonzero(
        [
            [False, True, False],
            [True, True, True],
            [True, False, True],
        ]
    ) = [
        [1, -1, -1],
        [0, 1, 2],
        [0, 2, -1],
    ]
    """
    assert tensor.dim() == 2 and tensor.dtype == torch.bool
    bsz = tensor.shape[0]
    max_per_batch = tensor.sum(-1).max()

    # batch_indices: [0, 0, 0, 1, 2, 3, 3, 5, 6]
    # per_batch_indices: [4, 6, 9, 2, 2, 5, 6, 9, 8]
    batch_indices, per_batch_indices = tensor.nonzero(as_tuple=True)
    # [0, 0, 1, 1, 1, 0, 2, 1]
    last_positions = batch_indices[1:] - batch_indices[:-1]
    assert (last_positions >= 0).all()  # i.e., batch_indices is sorted
    # [1, 1, 1, 2, 3, 4, 4, 6, 7]
    batch_indices_p1 = batch_indices + 1
    # [1, 2, 3, 5, 8, 12, 16, 22, 29]
    batch_indices_cumsum = batch_indices_p1.cumsum(0)
    # [0, 0, 3, 5, 8, 8, 16, 22]
    batch_base = (batch_indices_cumsum[:-1] * (last_positions > 0)).cummax(0).values
    # [1, 2, 3, 2, 3, 4, 8, 6, 7]
    batch_indices_cumsum[1:] -= batch_base
    assert (batch_indices_cumsum % batch_indices_p1 == 0).all()
    # [0, 1, 2, 0, 0, 0, 1, 0, 0]
    batch_arange = batch_indices_cumsum // batch_indices_p1 - 1

    nonzero_indices = per_batch_indices.new_full((bsz, max_per_batch), -1)
    nonzero_indices[batch_indices, batch_arange] = per_batch_indices
    return nonzero_indices


def lens_to_mask(lens, max_len=None):
    assert lens.dim() == 1
    if max_len is None:
        max_len = lens.max()
    return torch.arange(max_len, device=lens.device).expand(len(lens), -1) < lens.unsqueeze(1)


def get_sent_indices(tokens, sep_idx):
    # cat because we consider sep to belong to the previous sentence
    bsz = tokens.shape[0]
    return torch.cat((tokens.new_zeros(bsz, 1), (tokens == sep_idx).cumsum(1)[:, :-1]), dim=1)
