from typing import Protocol, Sequence, TypeVar
from ...typing import TensorLike

class HelmholtzPDEDataProtocol(Protocol):
    """Protocol interface for Helmholtz PDE data components.
    
    Defines the recommended protocol interface for Helmholtz-type partial differential equations,
    typically written as:
        Δu + k^2 u = f in Ω,
        u = g_D on ∂Ω_D,
        ∂u/∂n = g_N on ∂Ω_N.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry and dimension)
        2. PDE data terms (exact solution, gradient, Laplacian, source, wave number)
        3. Boundary condition terms (Dirichlet and Neumann)
        4. Additional utilities (boundary indicator functions)

    Notes:
        - Wave number `k` is central to Helmholtz equations.
        - The `laplacian` method is optional and can help for validation or source computation.
    """
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...

    def init_mesh(self): ...
    def wave_number(self) -> float: ...
    
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def laplacian(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...
    
    def neumann(self, p: TensorLike) -> TensorLike: ...
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike: ...

    def robin(self, p: TensorLike) -> TensorLike: ...
    def is_robin_boundary(self, p: TensorLike) -> TensorLike: ...

HelmholtzPDEDataT = TypeVar('HelmholtzPDEDataT', bound=HelmholtzPDEDataProtocol)

"""
DATA_TABLE is a registry for Helmholtz PDE models.
Each entry maps a model name to its corresponding module and class.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    "plane_wave": ("plane_wave_data_2d", "PlaneWaveData2D"),
    "bessel": ("bessel_data_2d", "BesselData2D"),
}

