from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike


class SingularGaussianData2D:
    """
    2D Quasilinear Poisson problem on a cut domain:

        -div(mu(|∇u|) ∇u) = f(x, y),  in Ω = (-1,1)^2 \ [0,1) × (-1,0]
                          u = 0      on ∂Ω

    with exact solution:
        u(x,y) = r^{2/3} * sin(2θ/3) + exp(-1000*(x-0.5)^2 - 1000*(y-0.5)^2)

    and nonlinear coefficient:
        mu(|∇u|) = 1 + exp(-|∇u|^2)
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
        polar_part = r**(bm.tensor(2.0) / 3.0) * bm.sin((2.0 * theta) / 3.0)
        gaussian = bm.exp(-1000.0 * ((x - 0.5)**2 + (y - 0.5)**2))
        return polar_part + gaussian

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        cos = bm.cos
        theta = bm.atan2(y, x)
        theta[theta<0] += 2*bm.pi
        exp = bm.exp
        u_x = 2*x*sin(2*theta/3)/(3*(x**2 + y**2)**(2/3)) - 2*y*cos(2*theta/3)/(3*(x**2 + y**2)**(2/3)) + (1000.0 - 2000*x)*exp(-1000*(x - 0.5)**2 - 1000*(y - 0.5)**2)
        u_y = 2*x*cos(2*theta/3)/(3*(x**2 + y**2)**(2/3)) + 2*y*sin(2*theta/3)/(3*(x**2 + y**2)**(2/3)) + (1000.0 - 2000*y)*exp(-1000*(x - 0.5)**2 - 1000*(y - 0.5)**2) 
        return bm.stack([u_x, u_y], axis=-1)

    @cartesian
    def nonlinear_coeff(self, p: TensorLike) -> TensorLike:
        grad = self.gradient(p)
        grad_norm_sq = bm.sum(grad**2, axis=-1)
        return 1.0 + bm.exp(-grad_norm_sq)


    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        r = bm.sqrt(x**2+y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] += 2*bm.pi
        E = bm.exp(-1000*(x - 0.5)**2 - 1000*(y - 0.5)**2)
        R23 = (x**2+y**2)**(2/3)
        R53 = (x**2+y**2)**(5/3)
        si = bm.sin(2*theta/3)
        co = bm.cos(2*theta/3)
        return 2*(R23**1.0*((2*(3*E*R23**1.0*(2000*x - 1000.0) + 2*co*y - 2*si*x)*(9000*E*R23**1.0*R53**1.0*(2000*(x - 0.5)**2 - 1) + 2*R23**1.0*(co*x*y - 2*si*x**2 - si*y**2) + 3*R53**1.0*si) + (-3*E*R23**1.0*(2000*y - 1000.0) + 2*co*x + 2*si*y)*(-9*E*R23**1.0*R53**1.0*(2000*x - 1000.0)*(2000*y - 1000.0) + 4*R23**1.0*(2*co*x**2 + co*y**2 + si*x*y) - 6*R53**1.0*co))*(3*E*R23**1.0*(2000*x - 1000.0) + 2*co*y - 2*si*x) - ((3*E*R23**1.0*(2000*x - 1000.0) + 2*co*y - 2*si*x)*(9*E*R23**1.0*R53**1.0*(2000*x - 1000.0)*(2000*y - 1000.0) + 4*R23**1.0*(co*x**2 + 2*co*y**2 - si*x*y) - 6*R53**1.0*co) + 2*(-3*E*R23**1.0*(2000*y - 1000.0) + 2*co*x + 2*si*y)*(9000*E*R23**1.0*R53**1.0*(1 - 2000*(y - 0.5)**2) + 2*R23**1.0*(co*x*y + si*x**2 + 2*si*y**2) - 3*R53**1.0*si))*(-3*E*R23**1.0*(2000*y - 1000.0) + 2*co*x + 2*si*y))*exp(-((3*E*R23**1.0*(1000.0 - 2000*x) - 2*co*y + 2*si*x)**2 + (3*E*R23**1.0*(1000.0 - 2000*y) + 2*co*x + 2*si*y)**2)/(9*R23**2.0)) + 18*R23**3.0*(1 + exp(-((-3*E*R23**1.0*(2000*x - 1000.0) - 2*co*y + 2*si*x)**2 + (-3*E*R23**1.0*(2000*y - 1000.0) + 2*co*x + 2*si*y)**2)/(9*R23**2.0)))*(4500*E*R23**1.0*R53**1.0*(1 - 2000*(y - 0.5)**2) - 4500*E*R23**1.0*R53**1.0*(2000*(x - 0.5)**2 - 1) - R23**1.0*(co*x*y - 2*si*x**2 - si*y**2) + R23**1.0*(co*x*y + si*x**2 + 2*si*y**2) - 3*R53**1.0*si))/(81*R23**4.0*R53**1.0)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        # 标准边界 + 截去的小区域 [0,1) x (-1,0]
        return (bm.abs(x + 1.0) < atol) | (bm.abs(x - 1.0) < atol) | \
               (bm.abs(y + 1.0) < atol) | (bm.abs(y - 1.0) < atol) | \
               ((x >= 0.0) & (x < 1.0) & (y >= -1.0) & (y < 0.0))

