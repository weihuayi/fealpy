from typing import (
    List,
    Tuple,
    Callable,
    Union,
    Protocol,
    Any
)

import numpy as np
from torch import Tensor
import numpy as np
from numpy.typing import NDArray

TensorOrArray = Union[Tensor, NDArray]

TensorFunction = Callable[[Tensor], Tensor]
VectorFunction = Callable[[NDArray], NDArray]
Operator = Callable[[Tensor, Tensor], Tensor]
Index = Union[int, bool, Tensor, slice, List, Tuple]
S: Index = slice(None, None, None)


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
    def entity_measure(self, etype='cell', index=np.s_[:]) -> Union[NDArray, float]: ...
