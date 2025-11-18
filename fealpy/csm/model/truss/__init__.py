from typing import Protocol, Sequence, TypeVar, overload,Optional

from fealpy.backend import TensorLike

class TrussPDEDataProtocol(Protocol):
    """

    A protocol for truss PDE data classes.This protocol defines the expected methods and properties
    that any Truss PDE data class should implement.

    Parameters:
        E (float): Young's modulus.
        A (float): Cross-sectional area.
        f (float): load.
        L (float): bar length.
        n (int): Number of mesh elements.
    Methods:
        init_mesh(p: TensorLike) -> None:
            Initialize the mesh for the beam domain.
        source(p: TensorLike) -> TensorLike:
            Compute the source term for the beam PDE.
        geo_dimension() -> int:
            Return the geometric dimension of the domain.
        domain() -> Sequence[float]:
            Return the computational domain [xmin, xmax].
    The protocol is designed to be implemented by specific bar PDE data classes.
    """

    def __init__(self, data: TensorLike) -> None: ...
    
    @property
    def data(self) -> TensorLike: ...
    
    @data.setter
    def data(self, value: TensorLike) -> None: ...
    def init_mesh(self, p: TensorLike) -> None: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def dirichlet(self, p: TensorLike) -> Optional[TensorLike]: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> Optional[TensorLike]: ...

TrussPDEDataT = TypeVar('TrussPDEDataT', bound=TrussPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # Add beam PDE models here (file_name, class_name)
    1: ("bar_data_1d", "BarData1D"),
    2: ("truss_data_2d", "TrussData2D"),
    3: ("truss_data_3d", "TrussData3D"),
    4: ("truss_tower_data_3d", "TrussTowerData3D")
}