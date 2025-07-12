from typing import Protocol, Sequence, TypeVar
from ...typing import TensorLike

class HelmholtzPDEDataProtocol(Protocol):
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

    def wave_number(self) -> float: ...
    
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def laplacian(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...
    
    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike:...
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike: ...

    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike: ...
    def is_robin_boundary(self, p: TensorLike) -> TensorLike: ...

HelmholtzPDEDataT = TypeVar('HelmholtzPDEDataT', bound=HelmholtzPDEDataProtocol)

"""
DATA_TABLE is a registry for Helmholtz PDE models.
Each entry maps a model name to its corresponding module and class.
"""
DATA_TABLE = {
        1: ("exp0001", "EXP0001"),
        2: ("exp0002", "EXP0002"),
        3: ("exp0003", "EXP0003"),
        4: ("exp0004", "EXP0004"),
        5: ("exp0005", "EXP0005")
}

