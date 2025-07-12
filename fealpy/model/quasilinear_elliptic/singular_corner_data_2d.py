from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike


class SingularCornerData2D:
    """
    2D quasilinear Poisson problem with singularity at the origin:

        -div(mu(x, |∇u|) ∇u) = f(x, y),  in Ω = (-1,1)^2 \ [0,1) × (-1,0]
                              u = 0      on ∂Ω

    Exact solution in polar coordinates:
        u(r, θ) = r^{2/3} * sin(2θ / 3)

    Nonlinear diffusion coefficient:
        mu(x, |∇u|) = 1 + exp(-|∇u|^2)
    """

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return [-1.0, 1.0, -1.0, 1.0]

    @cartesian
    def threshold(self, p) -> TensorLike:
        """ L domain"""
        x, y = p[..., 0], p[..., 1]
        return (x>=0) & (y<0) 

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] += 2*bm.pi
        return r**(2.0 / 3.0) * bm.sin((2.0 * theta) / 3.0)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        sin = bm.sin
        cos = bm.cos
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] += 2*bm.pi
        u_x = 2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*(x**2 + y**2)**(2/3))
        u_y = 2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*(x**2 + y**2)**(2/3))
        u_x = bm.where(r > 1e-14, u_x, 0.0)  
        u_y = bm.where(r > 1e-14, u_y, 0.0)
        return bm.stack((u_x, u_y), axis=-1)

    @cartesian
    def nonlinear_coeff(self, p: TensorLike) -> TensorLike:
        grad = self.gradient(p)
        grad_norm_sq = bm.sum(grad**2, axis=-1)
        return 1.0 + bm.exp(-grad_norm_sq)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        sin = bm.sin
        cos = bm.cos
        atan2 = bm.atan2
        theta = bm.atan2(y, x)
        theta[theta<0] += 2*bm.pi
        r = bm.sqrt(x**2 + y**2)
        f = -16*exp(-4/(9*(x**2 + y**2)**(1/3)))*sin(2*theta/3)/(81*r**2)
        f = bm.where(r > 1e-14, f, 0.0)
        return f

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (bm.abs(x + 1.0) < atol) | (bm.abs(x - 1.0) < atol) | \
               (bm.abs(y + 1.0) < atol) | (bm.abs(y - 1.0) < atol) | \
               ((x >= 0.0) & (x < 1.0) & (y > -1.0) & (y <= 0.0))  # cut-out region

