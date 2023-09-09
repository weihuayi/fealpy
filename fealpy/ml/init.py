
from typing import Sequence

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
