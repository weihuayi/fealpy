from typing import Protocol, Sequence, TypeVar
from fealpy.backend import TensorLike

class StokesPDEDataProtocol(Protocol):
    """Protocol interface for Navier-Stokes PDE data components with viscosity, convertion and source terms.
    
    Defines the recommended protocol interface for Navier-Stokes partial differential equation solvers.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry,computational domain)
        2. Equation terms methods (exact solution, grdient and source terms)
        3. Boundary condition methods (Dirichlet, Neumann, Robin types)

    Notes:  
        This protocol serves as a development guideline - implementing classes are encouraged to:
        - Provide implementations for the declared methods
        - Maintain consistent method signatures and return types
        - Implement methods relevant to their use case
    """
    def __str__(self) -> str: ...
    def get_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def init_mesh(self, nx: int = 8, ny: int = 8) -> TensorLike: ...
    def velocity(self, p: TensorLike) -> TensorLike: ...
    def pressure(self, p: TensorLike) -> TensorLike: ...
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike: ...
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike: ...
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...

StationaryNSPDEDataT = TypeVar('StationaryNSPDEDataT', bound=StokesPDEDataProtocol)
"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    1: ("exp0001", "Exp0001")
}