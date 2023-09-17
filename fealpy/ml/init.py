
from typing import Sequence, List, Union

import torch
from torch import Tensor

def fill_(tensor: Tensor, vals: Sequence[float], dim=0):
    """
    @brief Fills out the input `Tensor` with values repeatly on the given dimension.

    @param tensor: Tensor.
    @param vals: Sequence[float]. The sequence of data to fill out the tensor with.
    @param dim: int. The dimension along which the input is filled.
    """
    l = len(vals)
    ndim = tensor.ndim
    assert dim < ndim, "The given dim is out of the dimensions of tensor."
    idx = [slice(None, None, None), ] * ndim
    with torch.no_grad():
        for i, val in enumerate(vals):
            idx[dim] = slice(i, None, l)
            tensor[idx] = val
    return tensor


def cross_fill_(tensor: Tensor, *vals: Union[Sequence[float], Tensor]):
    """
    @brief Fills out the input `Tensor` with value combinations repeatly.

    @param tensor: Tensor.
    @param vals: Sequence[float]. Some float sequences. Values in these sequences\
           will be combined as the data to fill the tensor. The number of these\
           sequences must match the dimension of the tensor in the last dim/axis.
    """
    dim = len(vals)
    assert dim == tensor.shape[-1], "Number of value sequence does not match the last dimension of the input."
    val_tensors: List[Tensor] = []

    for vseq in vals:
        if isinstance(vseq, Tensor):
            val_tensors.append(vseq)
        else:
            val_tensors.append(torch.tensor(vseq, dtype=tensor.dtype, device=tensor.device))

    val_meshes = torch.meshgrid(val_tensors, indexing='ij')
    value = torch.stack(val_meshes, dim=-1).reshape(-1, dim)
    del val_tensors, val_meshes
    n_vals = value.shape[0]
    n_repeats = tensor.shape[0] // n_vals + 1
    temp = value.expand(n_repeats, n_vals, dim).reshape(-1, dim)
    with torch.no_grad():
        tensor[:] = temp[:tensor.shape[0], :]
    return tensor
