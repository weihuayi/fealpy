from typing import Optional
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d

class Exp0003(BoxMesher2d):
    """
    2D Linear Elasticity problem with polynomial displacement

    -∇·σ = b    in Ω
      Aσ = ε(u) in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)

    Material parameters:
    lam = 1, mu = 0.3

    For isotropic materials:
    Aσ = (1/2μ)σ - (λ/(2μ(dλ+2μ)))tr(σ)I

    """

    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(self.box)
        self.hypo = 'plane_strain' 

    def geo_dimension(self):
        return 2

    def lam(self, p: Optional[TensorLike] = None) -> float:
        return 1.0

    def mu(self, p: Optional[TensorLike] = None) -> float:
        return 0.5
    
    def stress_matrix_coefficient(self) -> tuple[float, float]:
        d = self.geo_dimension()
        lam = self.lam()
        mu = self.mu()
        lambda0 = 1.0 / (2 * mu)
        lambda1 = lam / (2 * mu * (d * lam + 2 * mu))

        return lambda0, lambda1
    
    @cartesian
    def displacement(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        exp_xy = bm.exp(x - y)

        u1 = exp_xy * x * (1 - x) * y * (1 - y)
        u2 = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

        val = bm.stack([u1, u2], axis=-1)

        return val
    
    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        exp_xy = bm.exp(x - y)

        b1_term1 = 2 * (x**2 + 3*x) * (y - y**2) * exp_xy
        b1_term2 = -0.5 * (x - x**2) * (-y**2 + 5*y - 4) * exp_xy
        b1_term3 = -1.5 * bm.pi**2 * bm.cos(bm.pi * x) * bm.cos(bm.pi * y)
        b1 = b1_term1 + b1_term2 + b1_term3

        b2_term1 = -1.5 * (1 - x - x**2) * (1 - 3*y + y**2) * exp_xy
        b2_term2 = 2.5 * bm.pi**2 * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        b2 = b2_term1 + b2_term2

        val = bm.stack([b1, b2], axis=-1)

        return  val
    
    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        return self.displacement(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        domain = self.domain()
        eps = 1e-12 
        x, y = p[..., 0], p[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < eps
        flag_x1 = bm.abs(x - domain[1]) < eps
        flag_y0 = bm.abs(y - domain[2]) < eps
        flag_y1 = bm.abs(y - domain[3]) < eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1
        
        return flag
    
    @cartesian
    def stress(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        exp_xy = bm.exp(x - y)
        pi = bm.pi
        
        du1_dx = exp_xy * y * (1 - y) * (1 - x - x**2) 
        du1_dy = exp_xy * x * (1 - x) * (1 - 3*y + y**2)
        du2_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du2_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        
        eps_xx = du1_dx
        eps_yy = du2_dy
        eps_xy = 0.5 * (du1_dy + du2_dx)
        
        lam, mu = self.lam(), self.mu()
        sigma_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sigma_yy = lam * (eps_xx + eps_yy) + 2 * mu * eps_yy
        sigma_xy = 2 * mu * eps_xy
        
        val = bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)
        
        return val