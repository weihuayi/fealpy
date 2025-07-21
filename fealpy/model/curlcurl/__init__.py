from typing import Protocol, Sequence, TypeVar
from ...typing import TensorLike

class CurlCurlPDEDataProtocol(Protocol):
    """
    Protocol interface for time-harmonic vector curl-curl PDEs (Maxwell-type equations).

    Defines the recommended protocol interface for equations of the form:

        curl curl E - k² E = f(x),      in Ω,
        with appropriate boundary conditions.

    where E is a vector-valued field (in ℝ² or ℝ³).

    This protocol suggests four main categories of methods that implementing classes may provide:

        1. Domain specification methods (geometry and dimension)
        2. PDE data terms (exact solution, curl, source, wave number)
        3. Boundary condition terms (Dirichlet, Neumann, Robin)
        4. Additional utilities (boundary indicator functions)

    Notes:
        - Wave number `k` is central to time-harmonic Maxwell equations.
        - All fields like `solution`, `source`, `curl`, should return vector-valued data with shape (N, d).
        - Boundary condition methods may raise NotImplementedError if unused.
    """
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...

    def wave_number(self) -> float: ...

    def solution(self, p: TensorLike) -> TensorLike: ...
    def curl(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...

    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike: ...
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike: ...

    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike: ...
    def is_robin_boundary(self, p: TensorLike) -> TensorLike: ...



CurlCurlPDEDataT = TypeVar('CurlCurlPDEDataT', bound=CurlCurlPDEDataProtocol)


DATA_TABLE = {
        1 : ("exp0001", "Exp0001")
}



