from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

class Exp0002(BoxMesher2d):
    """
    2D stationay incompressible Navier-Stokes problem for a Newtonian fluid:

        ρ[(u · ∇)u] = ∇·σ + ρ·f,       in Ω × (0, T]
              ∇·u = 0,                in Ω × (0, T]
               u = g_u,               on Γ_u × (0, T]
           u(x, 0) = u₀(x),           in Ω

    The Cauchy stress tensor σ is decomposed as:

        σ = τ - p·I

    where τ is the deviatoric (viscous) stress tensor given by:

        τ = 2μ·ε
        ε = 0.5·(∇u + ∇ᵗu)

    Physical variables:
        u(x, y)   : velocity vector field
        p(x, y)   : pressure scalar field
        ρ         : fluid density
        μ         : dynamic viscosity
        f(x, y)   : external body force
        g_u       : prescribed velocity on Dirichlet boundary

    Domain:
        Ω = (0, 1) × (0, 1)

    Exact (manufactured) solution:

        u₁(x, y) = (0.1 / 2π)·exp(0.1·y)/(exp(0.1)-1) · 
                   sin(2π·(exp(0.1·y) - 1)/(exp(0.1) - 1)) · 
                   (1 - cos(2π·(exp(3·x) - 1)/(exp(3) - 1)))

        u₂(x, y) = -(3 / 2π)·exp(3·x)/(exp(3)-1) · 
                   sin(2π·(exp(3·x) - 1)/(exp(3) - 1)) · 
                   (1 - cos(2π·(exp(0.1·y) - 1)/(exp(0.1) - 1)))

        p(x, y) = 0.3·exp(3·x)·exp(0.1·y)/[(exp(3)-1)(exp(0.1)-1)] · 
                  sin(2π·(exp(3·x) - 1)/(exp(3) - 1)) · 
                  sin(2π·(exp(0.1·y) - 1)/(exp(0.1) - 1)) · 
                  (1 - sin(2π·(exp(3·x) - 1)/(exp(3) - 1)))

    Properties:
        - Divergence-free velocity field
        - Pressure is smooth but nontrivial
        - Exponential mapping introduces strong local gradients

    Parameters:
        ρ = 1.0    # Fluid density
        μ = 1.0    # Dynamic viscosity

    This test case provides a more challenging benchmark than simple polynomials,
    as it contains nonlinearity in both exponential and trigonometric form.

    Reference:
        https://www.sciencedirect.com/science/article/abs/pii/S0045782502005133
    """
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        super().__init__(box=self.box)
        self.mesh = self.init_mesh[options.get('init_mesh', 'uniform_tri')](nx=options.get('nx', 8), ny=options.get('ny', 8))

    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the PDE configuration."""
        s = f"{self.__class__.__name__}(\n"
        s += f"  problem            : 2D stationary incompressible Navier-Stokes\n"
        s += f"  domain             : {self.box}\n"
        s += f"  mesh size          : nx = {self.nx}, ny = {self.ny}\n"
        s += f"  density (ρ)        : {self.rho}\n"
        s += f"  viscosity (μ)      : {self.mu}\n"
        s += f"  exact_velocity_x   : u_1(x, y) = 10·x²·(x - 1)²·y·(y - 1)·(2y - 1)\n"
        s += f"  exact_velocity_y   : u_2(x, y) = -10·x·(x - 1)·(2x - 1)·y²·(y - 1)²\n"
        s += f"  exact_pressure     : p(x, y) = 10·(2x - 1)·(2y - 1)\n"
        s += f")"
        return s

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        exp = bm.exp
        r1 = 3  
        r2 = 0.1  
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = (r2 / (2*pi)
                        * exp(r2 * y) / (exp(r2) - 1)
                        * bm.sin(2*pi * (exp(r2 * y) - 1) / (exp(r2) - 1))
                        * (1 - bm.cos(2*pi * (exp(r1 * x) - 1) / (exp(r1) - 1))))
        result[..., 1] = (-r1 / (2*pi)
                        * exp(r1 * x) / (exp(r1) - 1)
                        * bm.sin(2*pi * (exp(r1 * x) - 1) / (exp(r1) - 1))
                        * (1 - bm.cos(2*pi * (exp(r2 * y) - 1) / (exp(r2) - 1))))
        return result
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        exp = bm.exp
        r1 = 3  
        r2 = 0.1
        return (r1 * r2
                * exp(r1 * x) * exp(r2 * y)
                / ((exp(r1) - 1) * (exp(r2) - 1))
                * bm.sin(2*pi * (exp(r1 * x) - 1) / (exp(r1) - 1))
                * bm.sin(2*pi * (exp(r2 * y) - 1) / (exp(r2) - 1)))
    
    def pressure_integral_target(self) -> float:
        """Integral of the exact pressure over the domain."""
        return 0.0
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        exp = bm.exp
        cos = bm.cos
        sin = bm.sin
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = -1.5*(1 - cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10))))*((1 - cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3))))*exp(y/10)*sin(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))/(200*pi*(-1 + exp(1/10))) + (1 - cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3))))*exp(y/5)*cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))/(100*(-1 + exp(1/10))**2))*exp(3*x)*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/(pi*(-1 + exp(3))) + 0.015*(1 - cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3))))*exp(3*x)*exp(y/5)*sin(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))**2*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/(pi*(-1 + exp(1/10))**2*(-1 + exp(3))) + 0.002*pi*(1 - cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3))))*exp(3*y/10)*sin(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))/(-1 + exp(1/10))**3 - 0.0005*(1 - cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3))))*exp(y/10)*sin(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))/(pi*(-1 + exp(1/10))) - 0.003*(1 - cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3))))*exp(y/5)*cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))/(-1 + exp(1/10))**2
        result[..., 1] = -54.0*pi*(1 - cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10))))*exp(9*x)*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/(-1 + exp(3))**3 + 0.45*(1 - cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10))))*exp(6*x)*exp(y/10)*sin(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))**2/(pi*(-1 + exp(1/10))*(-1 + exp(3))**2) + 81.0*(1 - cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10))))*exp(6*x)*cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/(-1 + exp(3))**2 + 13.5*(1 - cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10))))*exp(3*x)*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/(pi*(-1 + exp(3))) + 0.05*(1 - cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3))))*(-9*(1 - cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10))))*exp(6*x)*cos(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/(-1 + exp(3))**2 - 9*(1 - cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10))))*exp(3*x)*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/(2*pi*(-1 + exp(3))))*exp(y/10)*sin(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))/(pi*(-1 + exp(1/10))) + 0.06*exp(3*x)*exp(y/10)*sin(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))/((-1 + exp(1/10))*(-1 + exp(3))) + 0.12*pi*exp(3*x)*exp(y/5)*sin(2*pi*(exp(3*x) - 1)/(-1 + exp(3)))*cos(2*pi*(exp(y/10) - 1)/(-1 + exp(1/10)))/((-1 + exp(1/10))**2*(-1 + exp(3)))
        return result

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        # result = bm.ones_like(p[..., 0], dtype=bm.bool)
        # return result
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        if p is None:
            return 0
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        return self.velocity(p)
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        return self.pressure(p)

