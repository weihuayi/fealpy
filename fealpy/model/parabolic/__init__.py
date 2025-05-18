from typing import Protocol, Sequence, TypeVar, overload,Optional
from ...backend import TensorLike

class ParabolicPDEDataProtocol(Protocol):
    """Protocol interface for parabolic PDE data components.
    
    Defines the recommended protocol interface for parabolic partial differential equation solvers.

    This protocol suggests four main categories of methods that implementing classes may provide:
    1. Domain specification methods (geometry, computational domain, duration)
    2. PDE coefficient methods (diffusion, convection, reaction terms)
    (Notes:When coefficients (diffusion, convection, reaction) are tensor-valued,
            the node coordinate tensor p can be omitted in method calls.)
    3. Equation terms methods (exact solution, grdient, init_solution and source terms)
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
    @overload
    def convection_coef(self, p: TensorLike) -> TensorLike: ...
    @overload
    def convection_coef(self) -> TensorLike: ...
    @overload
    def reaction_coef(self, p: TensorLike) -> TensorLike: ...
    @overload
    def reaction_coef(self) -> TensorLike: ...
    def duration(self) -> Sequence[float]: ...
    def init_solution(self, p: TensorLike) -> TensorLike: ...
    def solution(self, p: TensorLike, t: float) -> TensorLike: ...
    def gradient(self, p: TensorLike, t: float) -> TensorLike: ...
    def source(self, p: TensorLike, t: float) -> TensorLike: ...
    def dirichlet(self, p: TensorLike, t:float) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

ParabolicPDEDataT = TypeVar('ParabolicPDEDataT', bound=ParabolicPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    "sinexp": ("sin_exp_data_1d", "SinExpData1D"),
    "sinsinexp": ("sin_sin_exp_data_2d", "SinSinExpData2D"),

}


