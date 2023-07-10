from typing import (
    Callable,
    Optional,
    Union,
    Protocol,
    Any
)


from torch import Tensor
import numpy as np
from numpy.typing import NDArray

dtype = Optional[bool]
TensorOrArray = Union[Tensor, NDArray]

TensorFunction = Callable[[Tensor], Tensor]
VectorFunction = Callable[[NDArray], NDArray]
Operator = Callable[[Tensor, TensorFunction], Tensor]


class GeneralSampler(Protocol):
    """A protocol class for all samplers. This is not runtime-checkable."""
    @property
    def m(self) -> int: ...
    @property
    def nd(self) -> int: ...
    def run(self) -> Tensor: ...


class MeshLike(Protocol):
    """A simple protocal for meshes."""
    def entity(self, etype) -> Any: ...
