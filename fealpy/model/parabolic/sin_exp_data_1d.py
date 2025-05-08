from typing import Sequence
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinExpData1D:
    r"""
    One-dimensional parabolic equation:

        \underbrace{\frac{\partial u}{\partial t}}_{\text{Time derivative}}
        - \underbrace{\frac{\partial^2 u}{\partial x^2}}_{\text{Diffusion term}} 
        = \underbrace{-10 e^{-10t} \sin(4\pi x) + 16\pi^2 e^{-10t} \sin(4\pi x)}_{\text{Source term}}, 
        \quad x \in (0, 1),\ t > 0

        \underbrace{u(0,t) = u(1,t) = 0}_{\text{Dirichlet boundary conditions}}, 
        \quad t > 0

        \underbrace{u(x,0) = \sin(4\pi x)}_{\text{Initial condition}}

    Exact solution:
        u(x, t) = sin(4πx) · e^{-10t}

    Problem summary:
        - Geometry dimension: 1D
        - Domain: (0, 1)
        - Dirichlet BC at both ends
        - Suitable for verifying time-dependent solvers
    """

    def geo_dimension(self) -> int:
        return 1

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0]

    def solution(self, p: TensorLike, t: float) -> TensorLike:
        x = p
        return bm.sin(4 * bm.pi * x) * bm.exp(-10 * t)

    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        x = p
        return 4 * bm.pi * bm.cos(4 * bm.pi * x) * bm.exp(-10 * t)

    def source(self, p: TensorLike, t: float) -> TensorLike:
        x = p
        term1 = -10 * bm.exp(-10 * t) * bm.sin(4 * bm.pi * x)
        term2 = 16 * bm.pi ** 2 * bm.exp(-10 * t) * bm.sin(4 * bm.pi * x)
        return term1 + term2

    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        return self.solution(p, t)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return (bm.abs(p - 0.0) < 1e-12) | (bm.abs(p - 1.0) < 1e-12)

