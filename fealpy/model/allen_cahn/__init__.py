from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike

class AllenCahnPDEDataProtocol(Protocol):
    """Protocol interface for Allen-Cahn PDE data components.
    
    Defines the recommended protocol interface for Allen–Cahn type partial differential equations:
        φ_t + u·∇φ - ε² Δφ + f(φ) = 0

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry, computational domain)
        2. Equation terms methods (initial condition, double-well potential and its derivative, etc.)
        3. Boundary condition methods (Dirichlet, Neumann types)
        4. Model parameters (e.g., ε)

    Notes:
        - The method signatures assume that `p` is a tensor of shape (..., dim) representing points in space.
        - Time-dependent problems can be handled separately by the time integration module.

    """
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...

    def init_mesh(self): ...
    
    # Initial condition u0(x)
    def init_condition(self, p: TensorLike) -> TensorLike: ...
    
    # Double-well potential derivative f(u), e.g., f(φ) = 1/η^2 *(φ^3 - φ)
    def nonlinear_source(self, phi: TensorLike) -> TensorLike: ...
    
    # ε parameter in the Allen–Cahn equation
    def gamma(self) -> float: ...
    
    # Optional: exact solution if known (used for validation)
    def solution(self, p: TensorLike, t: float = 0.0) -> TensorLike: ...
    def gradient(self, p: TensorLike, t: float = 0.0) -> TensorLike: ...
    
    # Boundary condition specification
    def dirichlet(self, p: TensorLike, t: float = 0.0) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...
    def neumann(self, p: TensorLike, t: float = 0.0) -> TensorLike: ...
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike: ...

AllenCahnPDEDataT = TypeVar('AllenCahnPDEDataT', bound=AllenCahnPDEDataProtocol)

DATA_TABLE = {
    # example name: (file_name, class_name)
    1: ("exp0001", "Exp0001"),
    2: ("exp0002", "Exp0002"),
}

