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

    def init_mesh(self): ...
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
    "bessel": ("bessel_radiating_data_2d", "BesselRadiatingData2D"),
    "evanescent_wave": ("evanescent_wave_data_2d", "EvanescentWaveData2D"),
    "planewave_oblique_wave":("planeWave_oblique_incidence_data_2d", "PlaneWaveObliqueIncidenceData2D"),
    "sinsin2d": ("sinsin_data_2d", "SinsinData2D"),
    "sinsin3d": ("sinsin_data_3d", "SinsinData3D")
}

