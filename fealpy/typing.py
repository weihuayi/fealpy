
from typing import Tuple, Union, Literal, Callable
from functools import reduce

from .backend import TensorLike


Number = Union[int, float]
Index = Union[int, slice, Tuple[int, ...], TensorLike]
EntityName = Literal['cell', 'face', 'edge', 'node']
_int_func = Callable[..., int]
Barycenters = Union[Tuple[TensorLike, ...], TensorLike]


### Constants

_S = slice(None)


class Size(Tuple[int, ...]):
    def numel(self) -> int:
        return reduce(lambda x, y: x * y, self, 1)
