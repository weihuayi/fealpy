from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..mesher.box_mesher import BoxMesher2d
from ...decorator import cartesian


class Exp0001(BoxMesher2d):
    """
    2D Maxwell-type problem with complex Robin boundary condition:

        curl(curl(E)) - k^2 * E = f   in Ω = (0, 1)^2
        curl(E) cross n - i*k*E_t = g on ∂Ω

    Exact solution:
        E(x, y) = [x*y*(1 - x)*(1 - y), sin(πx) * sin(πy)]

    Boundary condition mimics absorbing (impedance-type) boundary.
    """

    def __init__(self, options: dict = {}):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)
        self.k = options.get('k', 1.0)

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        Fx = x*y*(1 - x)*(1 - y)
        Fy = bm.sin(bm.pi*x)*bm.sin(bm.pi*y)
        f = bm.concatenate([Fx, Fy], axis=-1) 
        return f 

    @cartesian
    def curl(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos

        curlF = -x*y*(x - 1) - x*(1 - x)*(1 - y) + pi*sin(pi*y)*cos(pi*x)
        return curlF
    
    @cartesian
    def curl_curl_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos

        ccFx = 2*x*(1 - x) + pi**2*cos(pi*x)*cos(pi*y)
        ccFy = x*y - x*(1 - y) - y*(1 - x) - (1 - y)*(x - 1) + pi**2*sin(pi*x)*sin(pi*y)
        ccf = bm.concatenate([ccFx, ccFy] , axis=-1)
        return ccf

    @cartesian
    def source(self, p):
        return self.curl_curl_solution(p) - self.k** 2 * self.solution(p)

    @cartesian
    def robin(self, p, n):
        t = bm.flip(n , axis=-1).copy()
        t[:, 0] = -t[:, 0]
        t = t[:, None, :]
        a = self.curl(p)[..., None] * t
        b = 1j * self.k * bm.einsum("eqd,eqd->eq", self.solution(p), t)[..., None] * t
        return a - b

    @cartesian
    def is_robin_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (
            (bm.abs(x - 0.0) < atol) |
            (bm.abs(x - 1.0) < atol) |
            (bm.abs(y - 0.0) < atol) |
            (bm.abs(y - 1.0) < atol)
        )
