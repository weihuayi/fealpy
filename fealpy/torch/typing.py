
from typing import Tuple, Union, Literal, Callable
from functools import reduce

import torch


### Types

Tensor = torch.Tensor
_dtype = torch.dtype
_device = torch.device

_bool = torch.bool
_int = torch.int
_int8 = torch.int8
_int16 = torch.int16
_int32 = torch.int32
_int64 = torch.int64
_float = torch.float
_float64 = torch.float16
_float32 = torch.float32
_float64 = torch.float64
_uint8 = torch.uint8

Number = Union[int, float]
DeviceLike = Union[str, _device]
Index = Union[int, slice, Tuple[int, ...], Tensor]
EntityName = Literal['cell', 'face', 'edge', 'node']
_int_func = Callable[..., int]
Barycenters = Union[Tuple[Tensor, ...], Tensor]


### Constants

_S = slice(None)


class Size(Tuple[int, ...]):
    def numel(self) -> int:
        return reduce(lambda x, y: x * y, self, 1)
