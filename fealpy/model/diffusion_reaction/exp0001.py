from typing import Sequence
from fealpy.decorator import cartesian, variantmethod
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from ..mesher.box_mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    Exp0001 provides data and methods for a 2D elliptic PDE problem with a exponential exact solution.
    The model problem is:
        -div(A ∇u) + c u = f,   in Ω = [0, 1] x [0, 1]
                ∇u · n = 0,        on ∂Ω (Neumann)
    with the exact solution:
        u(x, y) = cos(2πx)·cos(2πy)
    The diffusion coefficient A, reaction coefficient c, and source term f are defined as:
        A = [[10, 0], [0, 10]]
        c = 2
        f(x, y) = 2·cos(2πx)·cos(2πy) + 80π²·cos(2πx)·cos(2πy)
    Homogeneous Dirichlet or Neumann boundary conditions can be imposed on all boundaries.
    This class provides methods for mesh generation, coefficients, exact solution, gradient, flux, and boundary identification for use in finite element simulations.
    """
    """"Cosine-Cosine Solution Data"""
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2

    def domain(self):
        return self.box

    @cartesian
    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        """Diffusion coefficient"""
        val = bm.array([[10.0, 0.0], [0.0, 10.0]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)
    
    @cartesian
    def diffusion_coef_inv(self, p: TensorLike) -> TensorLike:
        """Inverse diffusion coefficient"""
        val = bm.array([[0.1, 0], [0, 0.1]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    @cartesian
    def reaction_coef(self, p: TensorLike) -> TensorLike:
        """Reaction coefficient"""
        val = bm.array([2.0])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Exact solution"""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.cos(2*pi*x)*bm.cos(2*pi*y)
        return val # val.shape == x.shape

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Gradient of the exact solution"""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -2*pi*bm.sin(2*pi*x)*bm.cos(2*pi*y),
            -2*pi*bm.cos(2*pi*x)*bm.sin(2*pi*y)), axis=-1)
        return val # val.shape == p.shape
    
    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        """Flux of the exact solution"""
        grad = self.gradient(p)
        val = self.diffusion_coef(p) 
        val = bm.einsum('...ij, ...j->...i', val, -grad)
        return val
    
    @cartesian
    def source(self, p: TensorLike, index=None) -> TensorLike:
        """Compute exact source"""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 2*bm.cos(2*pi*x)*bm.cos(2*pi*y) + 80*pi**2*bm.cos(2*pi*x)*bm.cos(2*pi*y)
        return val
    
    @cartesian
    def grad_dirichlet(self, p, space):
        """Gradient of the Dirichlet boundary condition."""
        return bm.zeros_like(p[..., 0])
    
    @cartesian
    def source1(self, p):
        """ Compute 0 source for a different form of the problem."""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 0*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)+1
        return val

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""        
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  # 绝对误差容限
        on_boundary = (
            (bm.abs(x - 1.) < atol) | (bm.abs(x) < atol) |
            (bm.abs(y - 1.) < atol) | (bm.abs(y) < atol)
        )
        return on_boundary 