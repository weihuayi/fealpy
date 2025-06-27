from typing import Protocol, Sequence, TypeVar,Optional, overload
from ...backend import TensorLike

class DiffusionReactionPDEDataProtocol(Protocol):
    """Protocol interface for Diffusion-Reaction PDE data components.
    
    Defines the recommended protocol interface for elliptic partial differential equation solvers.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry and computational domain)
        2. PDE coefficient methods (diffusion, convection, reaction terms)
        (Notes:When coefficients (diffusion, convection, reaction) are tensor-valued,
                the node coordinate tensor p can be omitted in method calls.)
        3. Equation terms methods (exact solution, grdient, flux and source terms)
        4. Boundary condition methods (Dirichlet, Neumann, Robin types)

    Notes:  
        This protocol serves as a development guideline - implementing classes are encouraged to:
        - Provide implementations for the declared methods
        - Maintain consistent method signatures and return types
        - Implement methods relevant to their use case
    """
    diffusion_coef 
    reaction_coef
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def flux(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

DiffusionReactionPDEDataT = TypeVar('DiffusionReactionPDEDataT', bound=DiffusionReactionPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    "coscos": ("cos_cos_data_2d", "CosCosData2D"),
    "sinsin": ("sin_sin_data_2d", "SinSinData2D"),
}
