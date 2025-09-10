from typing import Protocol, Sequence, TypeVar
from ...typing import TensorLike

class DLDMicrofluidicChipPDEDataProtocol(Protocol):
    """Protocol interface for Helmholtz PDE data components.
    
    Defines the recommended protocol interface for time-harmonic wave equations,
    typically written as:
            -Δu - k²u = f(x)  in Ω
            with appropriate boundary conditions.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry and dimension)
        2. PDE data terms (exact solution, gradient, Laplacian, source, wave number)
        3. Boundary condition terms (Dirichlet, Neumann, Robin)
        4. Additional utilities (boundary indicator functions)

    Notes:
        - Wave number `k` is central to Helmholtz equations.
        - The `laplacian` method is optional and can help for validation or source computation.
    """
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike: ...
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike: ...
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike: ...

HelmholtzPDEDataT = TypeVar('DLDMicrofluidicChipPDEDataT', bound=DLDMicrofluidicChipPDEDataProtocol)

"""
DATA_TABLE is a registry for Helmholtz PDE models.
Each entry maps a model name to its corresponding module and class.
"""
DATA_TABLE = {
        1: ("exp0001", "Exp0001"),
        2: ("exp0002", "Exp0002")
}

