from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike

class QuasilinearEllipticPDEDataProtocol(Protocol):
    """Protocol interface for quasilinear elliptic PDE data components.
    
    Defines the recommended protocol interface for quasilinear elliptic partial differential equation solvers.

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
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def threshold(self, p:TensorLike) -> TensorLike: ...
    def nonlinear_coeff(self) -> TensorLike: ...
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

QuasilinearEllipticPDEDataT = TypeVar('QuasilinearEllipticPDEDataT',
                                      bound=QuasilinearEllipticPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    "sinsin": ("sin_sin_data_2d", "SinSinData2D"),
    "singular": ("singular_corner_data_2d", "SingularCornerData2D"),
    "singulargauss": ("singular_gaussian_data_2d", "SingularGaussianData2D"),
}
