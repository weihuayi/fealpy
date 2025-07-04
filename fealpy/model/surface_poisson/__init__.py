from typing import Protocol, overload, TypeVar
from ...typing import TensorLike

class SurfaceLeveLSetPDEDataProtocol(Protocol):
    """
    Protocol interface for PDE problems defined on level-set surfaces.

    This protocol defines a standard interface for PDEs posed on implicit
    surfaces (zero-level sets of signed distance functions). It includes
    methods for geometry representation, exact solution, source term,
    and boundary conditions.

    Tensor conventions:
        - p: coordinate tensor, shape (..., 3)
        - solution(p): (...,) or (..., C) depending on scalar/vector field
        - gradient(p): (..., 3) or (..., C, 3)
        - source(p): (...,)
        - exact(p): same shape as solution
        - normal(p): (..., 3)
        - hessian(p): (..., 3, 3)
    """

    def init_mesh(self): ...

    @overload
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient_of_solution(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...