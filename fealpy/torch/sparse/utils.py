
from typing import Optional

import torch

_Size = torch.Size
Tensor = torch.Tensor


def _dense_shape(values: Optional[Tensor]):
    if values is None:
        return _Size()
    else:
        return values.shape[:-1]


def _dense_ndim(values: Optional[Tensor]):
    if values is None:
        return 0
    else:
        return values.ndim - 1


def shape_to_strides(shape: _Size, item_size: int):
    strides = [item_size, ]

    for i in range(len(shape) - 1):
        strides.append(strides[-1] * shape[-i])

    return tuple(reversed(strides))
