
from typing import Callable, Dict, Literal, Sequence

from ...backend import TensorLike 
from ...backend import backend_manager as bm

from ..boundary_condition import BoundaryCondition, bc_mask, bc_value
from ...decorator import cartesian

class CosCosData():
    description = ""

    def __init__(self, bcs: Sequence[BoundaryCondition]=None):
        if bcs is None:
            bcs = [
                    BoundaryCondition(
                        mask_fn = lambda p: bm.logical_or(
                            bm.abs(p[:, 0] - 0.0) < 1e-12,
                            bm.abs(p[:, 0] - 1.0) < 1e-12,
                            bm.abs(p[:, 1] - 0.0) < 1e-12,
                            bm.abs(p[:, 1] - 1.0) < 1e-12,
                        ),
                        kind = 'dirichlet',
                        value_fn = self.solution
                        )
                    ]
        self.bcs = bcs

    def geo_dimension(self) -> int:
        return 2

    def domain(self):
        return [0., 1., 0., 1.]

    @cartesian
    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        val = bm.array([[10.0, 0.0], [0.0, 10.0]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)
    
    @cartesian
    def diffusion_coef_inv(self, p: TensorLike) -> TensorLike:
        val = bm.array([[0.1, 0], [0, 0.1]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    @cartesian
    def reaction_coef(self, p: TensorLike) -> TensorLike:
        val = bm.array([2.0])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.cos(2*pi*x)*bm.cos(2*pi*y)
        return val # val.shape == x.shape

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -2*pi*bm.sin(2*pi*x)*bm.cos(2*pi*y),
            -2*pi*bm.cos(2*pi*x)*bm.sin(2*pi*y)), axis=-1)
        return val # val.shape == p.shape
    
    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        grad = self.gradient(p)
        val = self.diffusion_coef(p) 
        val = bm.einsum('...ij, ...j->...i', val, -grad)
        return val
    
    @cartesian
    def source(self, p: TensorLike, index=None) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 2*bm.cos(2*pi*x)*bm.cos(2*pi*y) + 80*pi**2*bm.cos(2*pi*x)*bm.cos(2*pi*y)
        return val
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        return bc_value(p, self.bcs, 'dirichlet')
    
    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return bc_mask(p, self.bcs, 'dirichlet')
    
    @cartesian
    def neumann(self, p: TensorLike) -> TensorLike:
        return bc_value(p, self.bcs, 'neumann')
    
    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        return bc_mask(p, self.bcs, 'neumann')
    
    @cartesian
    def grad_dirichlet(self, p, space):
        return bm.zeros_like(p[..., 0])
    
    @cartesian
    def source1(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 0*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)+1
        return val

