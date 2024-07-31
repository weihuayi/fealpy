import torch
import builtins
from typing import Tuple, Union, Literal, Callable, Dict
from functools import reduce

from .backend import TensorLike

### Types
Tensor = torch.Tensor
Number = Union[builtins.int, builtins.float]
Index = Union[int, slice, Tuple[int, ...], TensorLike]
EntityName = Literal['cell', 'face', 'edge', 'node']
_int_func = Callable[..., int]
Barycenters = Union[Tuple[TensorLike, ...], TensorLike]
CoefLike = Union[Number, TensorLike, Callable[..., TensorLike]]


### Constants

_S = slice(None)


class Size(Tuple[int, ...]):
    def numel(self) -> int:
        return reduce(lambda x, y: x * y, self, 1)
