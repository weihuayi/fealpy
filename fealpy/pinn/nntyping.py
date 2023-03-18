from typing import (
    Callable,
    Optional,
    Union,
    Protocol,
    runtime_checkable
)


from torch import Tensor
from numpy.typing import NDArray

from ..mesh.Mesh2d import Mesh2d
from ..mesh.Mesh3d import Mesh3d

dtype = Optional[bool]
TensorOrArray = Union[Tensor, NDArray]
Mesh = Union[Mesh2d, Mesh3d]

TensorFunction = Callable[[Tensor], Tensor]
VectorFunction = Callable[[NDArray], NDArray]
Operator = Callable[[Tensor, TensorFunction], Tensor]


@runtime_checkable
class GeneralSampler(Protocol):
    """A protocol class for all samplers."""
    m: int
    nd: int
    def run(self) -> Tensor: ...
