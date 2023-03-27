from typing import (
    Callable,
    Optional,
    Union,
    Protocol,
    runtime_checkable
)


from torch import Tensor
import numpy as np
from numpy.typing import NDArray

from ..quadrature.Quadrature import Quadrature

dtype = Optional[bool]
TensorOrArray = Union[Tensor, NDArray]
# Mesh = Union[Mesh2d, Mesh3d]

TensorFunction = Callable[[Tensor], Tensor]
VectorFunction = Callable[[NDArray], NDArray]
Operator = Callable[[Tensor, TensorFunction], Tensor]


@runtime_checkable
class GeneralSampler(Protocol):
    """A protocol class for all samplers."""
    @property
    def m(self) -> int: ...
    @property
    def nd(self) -> int: ...
    def run(self) -> Tensor: ...


@runtime_checkable
class MeshLike(Protocol):
    def geo_dimension(self) -> int: ...
    def top_dimension(self) -> int: ...

    def entity(self, etype: Union[str, int]='cell') -> NDArray: ...
    def entity_measure(self, etype: Union[str, int]=3, index=np.s_[:]) -> NDArray: ...
    def entity_barycenter(self, etype: Union[str, int]='cell', index=np.s_[:]) ->NDArray: ...

    def integrator(self, q: int, etype: Union[int, str]) -> Quadrature: ...
    def cell_bc_to_point(self, bc: NDArray) -> NDArray: ...
