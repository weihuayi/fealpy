from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike

class PolyharmonicPDEDataProtocol(Protocol):
    """Protocol interface for polyharmonic PDE data components.
    
    The polyharmonic equation has the general form:
        (-Δ)^m u = f   in Ω
    
    This protocol defines a standard interface for:
        1. Domain and geometric dimension
        2. Source term and analytical solution (optional)
        3. Boundary conditions for high-order derivatives
        4. The order of the polyharmonic operator (m)

    Notes:
        - Coordinates `p` are assumed to be tensors of shape (..., dim)
        - The boundary condition interface supports specifying up to m-th order normal derivatives

    """
    
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    
    # Return the order m of the polyharmonic operator
    def order(self) -> int: ...
    
    # Right-hand side source term f(x)
    def source(self, p: TensorLike) -> TensorLike: ...
    
    # Optional: exact solution u(x) and its Laplacian powers
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def laplacian_k(self, p: TensorLike, k: int) -> TensorLike: ...
    
    # High-order Dirichlet boundary condition values: ∂^k u / ∂n^k = g_k(x), for k = 0,...,m-1
    def dirichlet_bc(self, p: TensorLike, k: int) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike, k: int) -> TensorLike: ...

PolyharmonicPDEDataT = TypeVar('PolyharmonicPDEDataT', bound=PolyharmonicPDEDataProtocol)

DATA_TABLE = {
    #example name: (file_name, class_name)
    1: ("exp0001", "Exp0001"),
    2: ("exp0002", "Exp0002"),
    3: ("exp0003", "Exp0003"),
    4: ("exp0004", "Exp0004")
}