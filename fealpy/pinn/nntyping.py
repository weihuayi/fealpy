from typing import (
    Callable,
    Optional,
    Union,
    Protocol,
    TypeVar,
    Literal
)


from torch import Tensor
import numpy as np
from numpy.typing import NDArray

dtype = Optional[bool]
TensorOrArray = Union[Tensor, NDArray]
# Mesh = Union[Mesh2d, Mesh3d]

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

_GD = TypeVar('_GD', bound=int, covariant=True)
_TD = TypeVar('_TD', bound=int, covariant=True)


class MeshLike(Protocol[_GD, _TD]):
    def geo_dimension(self) -> _GD: ...
    def top_dimension(self) -> _TD: ...

    def entity(self, etype: Union[str, int]='cell') -> NDArray: ...
    def entity_measure(self, etype: Union[str, int]=3, index=np.s_[:]) -> NDArray: ...
    def entity_barycenter(self, etype: Union[str, int]='cell', index=np.s_[:]) ->NDArray: ...

    def integrator(self, q: int, etype: Union[int, str]): ...
    def cell_bc_to_point(self, bc: NDArray) -> NDArray: ...


TriangleMesh = MeshLike[Literal[2], Literal[2]]
TetrahedronMesh = MeshLike[Literal[3], Literal[3]]
