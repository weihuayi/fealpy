from typing import (
    Callable,
    Optional,
    Union,
    Protocol,
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
    def geo_dimension(self) -> int: ...
    def top_dimension(self) -> int: ...
    def number_of_nodes_of_cells(self) -> int: ...

    def entity(self, etype: Union[str, int]='cell') -> NDArray: ...
    def entity_measure(self, etype: Union[str, int]=3, index=np.s_[:]) -> NDArray: ...
    def entity_barycenter(self, etype: Union[str, int]='cell', index=np.s_[:]) ->NDArray: ...

    def integrator(self, q: int, etype: Union[int, str]): ...
    def cell_bc_to_point(self, bc: NDArray) -> NDArray: ...
