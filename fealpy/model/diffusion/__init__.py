
from typing import Protocol, Sequence, TypeVar,Optional, overload
from ...backend import TensorLike

class DiffusionPDEDataProtocol(Protocol):
    """Protocol interface for elliptic PDE data components with only diffusion terms.
    
    Defines the recommended protocol interface for elliptic partial differential equation solvers.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry and computational domain)
        2. PDE coefficient methods (diffusion terms)
        (Notes:When coefficients (diffusion) are tensor-valued,
                the node coordinate tensor p can be omitted in method calls.)
        3. Equation terms methods (exact solution, grdient, flux and source terms)
        4. Boundary condition methods (Dirichlet, Neumann, Robin types)

    Notes:  
        This protocol serves as a development guideline - implementing classes are encouraged to:
        - Provide implementations for the declared methods
        - Maintain consistent method signatures and return types
        - Implement methods relevant to their use case
    """
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    @overload
    def diffusion_coef(self, p: Optional[TensorLike]) -> TensorLike: ...
    @overload
    def diffusion_coef(self) -> TensorLike: ...
    def diffusion_coef_inv(self, p: Optional[TensorLike] = None) -> TensorLike: ...
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def flux(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...
    def neumann(self, p: TensorLike) -> TensorLike: ...
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike: ...
    def robin(self, p: TensorLike) -> TensorLike: ...
    def is_robin_boundary(self, p: TensorLike) -> TensorLike: ...

DiffusionPDEDataT = TypeVar('DiffusionPDEDataT', bound=DiffusionPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
 
}
