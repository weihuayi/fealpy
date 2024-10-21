
import builtins
from typing import Tuple, Union, Literal, Callable

from .backend import TensorLike


### Types

_int_func = Callable[..., builtins.int]
Number = Union[builtins.int, builtins.float]
Scalar = Union[Number, TensorLike]
Index = Union[int, slice, Tuple[int, ...], TensorLike]
EntityName = Literal['cell', 'face', 'edge', 'node']
Barycenters = Union[Tuple[TensorLike, ...], TensorLike]
CoefLike = Union[Number, TensorLike, Callable[..., TensorLike]]
SourceLike = CoefLike
Threshold = Union[TensorLike, Callable[..., TensorLike]]


### Constants

_S = slice(None)
Size = Tuple[int, ...]
