
from typing import Tuple
from functools import reduce

import torch


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


class Size(Tuple[int, ...]):
    def numel(self) -> int:
        return reduce(lambda x, y: x * y, self, 1)
