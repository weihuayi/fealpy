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


class BesselRadiatingData2D(BoxDomainMesher2d):
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

    def set(self, k=1.0):
        self.k = bm.tensor(k, dtype=bm.float64)
        c1 = bm.cos(self.k) + bm.sin(self.k) * 1j
        c2 = bessel_function(0, self.k) + 1j * bessel_function(1, self.k)
        self.c = c1 / c2

    def geo_dimension(self) -> int:
        """Return the spatial dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the bounding box [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """
        Exact solution u(x, y) = (cos(k·r) - c·J0(k·r)) / k
        """
        if bm.backend_name == 'pytorch':
            x = p[..., 0:1]
            y = p[..., 1:2]
        elif bm.backend_name == 'numpy':
            x = p[..., 0]
            y = p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        val = bm.zeros(x.shape, dtype=bm.complex128)
        val[:] = (bm.cos(self.k * r) - self.c * bessel_function(0, self.k * r)) / self.k
        return val

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """
        Right-hand side: f(x, y) = sin(k·r)/r
        """
        x, y = p[..., 0:1], p[..., 1:2]
        r = bm.sqrt(x**2 + y**2)
        f = bm.zeros(x.shape, dtype=bm.complex128)
        f[:] = bm.sin(self.k * r) / r
        return f

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Gradient ∇u = u_r * (x/r, y/r), 
        where u_r = -sin(k·r) + c·J1(k·r)
        """
        x, y = p[..., 0:1], p[..., 1:2]
        r = bm.sqrt(x**2 + y**2)
        u_r = self.c * bessel_function(1, self.k * r) - bm.sin(self.k * r)

        val = bm.zeros(p.shape, dtype=bm.complex128)
        val[..., 0:1] = u_r * x / r
        val[..., 1:2] = u_r * y / r
        return val

    @cartesian
    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """
        Robin boundary data: g = ∂u/∂n + i·k·u
        """
        kappa = 1j * self.k
        if bm.backend_name == 'pytorch':
            val = (self.gradient(p) * n).sum(dim=-1, keepdim=True) + kappa * self.solution(p)
        elif bm.backend_name == 'numpy':
            grad = self.gradient(p)
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

