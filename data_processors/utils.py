import torch
import torch.nn.functional as F


def flatten(l):
    return [e for subl in l for e in subl]


def pad(tensors, pad_idx, no_pad_dim=None, left_pad=False):
    dim = tensors[0].dim()
    assert all(tensor.dim() == dim for tensor in tensors)
    max_shape = [-1] * dim
    for tensor in tensors:
        for i, s in enumerate(tensor.shape):
            max_shape[i] = max(max_shape[i], s)

    padded_tensors = []
    for tensor in tensors:
        pad = []
        for d, (m, s) in enumerate(zip(max_shape, tensor.shape)):
            if no_pad_dim is not None and d == no_pad_dim:
                pad.append([0, 0])
            else:
                pad.append(([0] if not left_pad else []) + [m - s] + ([0] if left_pad else []))
        pad = flatten(pad[::-1])
        padded_tensors.append(F.pad(tensor, pad, value=pad_idx))

    return padded_tensors


def pad_and_stack(tensors, pad_idx, left_pad=False, default_dtype=torch.long):
    if len(tensors) == 0:  # TODO: the device here might be bad
        return torch.empty(0, 0, dtype=default_dtype)

    return torch.stack(pad(tensors, pad_idx, left_pad=left_pad), dim=0)


def get_sent_pairs(sent_level_dataset, start, src_end, use_sep, tgt_end=None):
    if tgt_end is None:
        tgt_end = src_end
    sent_pairs = [sent_level_dataset[i] for i in range(start, max(src_end, tgt_end) + 1)]
    sources = []
    targets = []
    # Remove eos or replace with sep from all context
    for side, l, end in (("source", sources, src_end), ("target", targets, tgt_end)):
        for i, sent_pair in enumerate(sent_pairs):
            assert sent_pair[side][-1].item() == sent_level_dataset.eos
            l.append(sent_pair[side].clone())
            if i == end - start:
                break
            if use_sep:
                # We checked that the sep index is the same across src/tgt
                l[-1][-1] = sent_level_dataset.src_dict.sep()
            else:
                l[-1] = l[-1][:-1]
        assert i == end - start
    return sources, targets
