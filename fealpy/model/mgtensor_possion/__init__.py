from typing import Protocol, Sequence, TypeVar
from ...typing import TensorLike

class MGTensorPossionPDEDataProtocol(Protocol):
    """Protocol interface for MGTensor Possion PDE data components.
    
    Defines the recommended protocol interface for poisson partial differential equation solvers.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry,computational domain)
        2. Equation terms methods (exact solution, grdient and source terms)
        3. Boundary condition methods (Dirichlet, Neumann, Robin types)
        4. scaling_function is a function that satisfies the boundary conditions.
    """
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

MGTensorPossionPDEDataT = TypeVar('MGTensorPossionPDEDataT', bound=MGTensorPossionPDEDataProtocol)

"""
DATA_TABLE is a registry for MGTensor Possion PDE models.
Each entry maps a model name to its corresponding module and class.
"""
DATA_TABLE = {
        1: ("exp0001", "Exp0001")

}

