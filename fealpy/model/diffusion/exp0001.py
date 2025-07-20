from typing import Sequence
from fealpy.decorator import cartesian, variantmethod
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from ..mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    PDE model class for vector-valued variable coefficient diffusion equation.
    
    Strong form:
        -∇·(ρ(x)∇u) = f,   in Ω = [0, 1] x [0, 1]
        u = 0,              on ∂Ω (homogeneous Dirichlet boundary condition)
    
    where:
        u(x,y) = [u1(x,y), u2(x,y)]^T is a 2D displacement vector
        f(x,y) = [f1(x,y), f2(x,y)]^T is a 2D source term vector  
        ρ(x,y) is a scalar diffusion coefficient
    
    Exact solution:
        u1(x,y) = sin(πx)sin(πy)
        u2(x,y) = sin(2πx)sin(2πy)
    
    Diffusion coefficient:
        ρ(x,y) = 1 + x + y
    
    Source terms:
        f1 = -π[cos(πx)sin(πy) + sin(πx)cos(πy)] + 2π²(1+x+y)sin(πx)sin(πy)
        f2 = -2π[cos(2πx)sin(2πy) + sin(2πx)cos(2πy)] + 8π²(1+x+y)sin(2πx)sin(2πy)
    """
    
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2
    
    def space_dimension(self) -> int:
        return 2

    def domain(self):
        return self.box

    @cartesian
    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        val = 1.0 + x + y
        return val

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        
        u1 = bm.sin(pi * x) * bm.sin(pi * y)
        u2 = bm.sin(2 * pi * x) * bm.sin(2 * pi * y)
        
        val = bm.stack((u1, u2), axis=-1)
        return val 

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        
        grad_u1_x = pi * bm.cos(pi * x) * bm.sin(pi * y)
        grad_u1_y = pi * bm.sin(pi * x) * bm.cos(pi * y)
        
        grad_u2_x = 2 * pi * bm.cos(2 * pi * x) * bm.sin(2 * pi * y)
        grad_u2_y = 2 * pi * bm.sin(2 * pi * x) * bm.cos(2 * pi * y)
        
        grad = bm.stack([
            bm.stack([grad_u1_x, grad_u1_y], axis=-1),
            bm.stack([grad_u2_x, grad_u2_y], axis=-1)
        ], axis=-2)
        
        return grad  

    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        grad = self.gradient(p)  
        rho = self.diffusion_coef(p)  
        
        rho_expanded = rho[..., None, None]
        
        val = -rho_expanded * grad
        return val  

    @cartesian
    def source(self, p: TensorLike, index=None) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        
        term1_1 = -pi * (bm.cos(pi * x) * bm.sin(pi * y) + 
                         bm.sin(pi * x) * bm.cos(pi * y))
        term1_2 = 2 * pi**2 * (1 + x + y) * bm.sin(pi * x) * bm.sin(pi * y)
        f1 = term1_1 + term1_2
        
        term2_1 = -2 * pi * (bm.cos(2 * pi * x) * bm.sin(2 * pi * y) + 
                             bm.sin(2 * pi * x) * bm.cos(2 * pi * y))
        term2_2 = 8 * pi**2 * (1 + x + y) * bm.sin(2 * pi * x) * bm.sin(2 * pi * y)
        f2 = term2_1 + term2_2
        
        val = bm.stack((f1, f2), axis=-1)

        return val
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        shape = p.shape[:-1] + (2,)
        return bm.zeros(shape)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  
        on_boundary = (
            (bm.abs(x - 1.0) < atol) | (bm.abs(x) < atol) |
            (bm.abs(y - 1.0) < atol) | (bm.abs(y) < atol)
        )

        return on_boundary