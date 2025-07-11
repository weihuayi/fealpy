from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..box_domain_mesher import BoxDomainMesher2d
from ...decorator import cartesian

def bessel_function(v: int, x: TensorLike) -> TensorLike:
    if bm.backend_name == 'pytorch':
        import torch
        if v == 0:
            return torch.special.bessel_j0(x)
        elif v == 1:
            return torch.special.bessel_j1(x)
        else:
            raise NotImplementedError("Only order 0 and 1 supported.")
    else:
        from scipy.special import jv
        if v == 0:
            return jv(0, x)
        elif v == 1:
            return jv(1, x)
        else:
            raise NotImplementedError("Just supports Bessel functions of order 0 and 1.")


class EXP0001(BoxDomainMesher2d):
    """
    2D Helmholtz problem with complex Robin boundary conditions:
    
        -Δu - k^2 u = f   in Ω = [0, 1]^2
         iku + ∂u/∂n = g  on ∂Ω

    Exact solution:
        u(x, y) = (cos(k·r) - c·J0(k·r)) / k
        where r = sqrt(x^2 + y^2), c = (cos(k) + i·sin(k)) / (J0(k) + i·J1(k))

    Source:
        f(x, y) = sin(k·r)/r

    Robin boundary term:
        g(x, y) = ∂u/∂n + i·k·u

    Source:
        https://cz5waila03cyo0tux1owpyofgoryroob.aminer.cn/A5/1A/1D/A51A1DBD4CE1D183344F2A280C430074.pdf
    """
    
    def __init__(self, option=None):
        self.box = [-0.5, 0.5, -0.5, 0.5]
        super().__init__(box=self.box)
        k = option.get('k', 1.0)
        self.k = bm.tensor(k, dtype=bm.float64)
        c1 = bm.cos(self.k) + bm.sin(self.k) * 1j
        c2 = bessel_function(0, self.k) + 1j * bessel_function(1, self.k)
        self.c = c1 / c2

    def geo_dimension(self) -> int:
        """Return the spatial dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the bounding box [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """
        Exact solution u(x, y) = (cos(k·r) - c·J0(k·r)) / k
        """
        x = p[..., 0]
        y = p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        return (bm.cos(self.k * r) - self.c * bessel_function(0, self.k * r)) / self.k

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """
        Right-hand side: f(x, y) = sin(k·r)/r
        """
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        return bm.sin(self.k * r) / r

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Gradient ∇u = u_r * (x/r, y/r), 
        where u_r = -sin(k·r) + c·J1(k·r)
        """
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        u_r = self.c * bessel_function(1, self.k * r) - bm.sin(self.k * r)

        du_dx = u_r * x / r
        du_dy = u_r * y / r
        return bm.stack((du_dx, du_dy), axis=-1)


    @cartesian
    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """
        Robin boundary data: g = ∂u/∂n + i·k·u
        """
        kappa = 1j * self.k
        grad = self.gradient(p)
        if len(grad.shape) == self.geo_dimension():
            val = bm.sum(grad * n, axis=-1)
        else:
            val = bm.sum(grad * n[:, None, :], axis=-1)
        val += kappa * self.solution(p)
        
        return val

    @cartesian
    def is_robin_boundary(self, p: TensorLike) -> TensorLike:
        """
        Mark boundary points: |x| = 0.5 or |y| = 0.5
        """
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (
            (bm.abs(x + 0.5) < atol) |
            (bm.abs(x - 0.5) < atol) |
            (bm.abs(y + 0.5) < atol) |
            (bm.abs(y - 0.5) < atol)
        )

