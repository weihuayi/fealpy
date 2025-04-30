
from typing import Callable, Dict, Literal, Sequence

from ...backend import TensorLike 
from ...backend import backend_manager as bm

from ..boundary_condition import BoundaryCondition, bc_mask, bc_value


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

    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        val = bm.array([[10.0, 1.0], [1.0, 10.0]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    def diffusion_coef_inv(self, p: TensorLike) -> TensorLike:
        val = bm.array([[0.1, -0.01], [-0.01, 0.1]])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    def reaction_coef(self, p: TensorLike) -> TensorLike:
        val = bm.array([2.0])
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)


    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.cos(2*pi*x)*bm.cos(2*pi*y)
        return val # val.shape == x.shape

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -2*pi*bm.sin(2*pi*x)*bm.cos(2*pi*y),
            -2*pi*bm.cos(2*pi*x)*bm.sin(2*pi*y)), axis=-1)
        return val # val.shape == p.shape

    def flux(self, p: TensorLike) -> TensorLike:
        grad = self.gradient(p)
        val = self.diffusion_coef(p) 
        val = bm.einsum('...ij, ...j->...i', val, -grad)
        return val

    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = -8*pi**2*sin(2*pi*x)*sin(2*pi*y) - 2*pi*sin(2*pi*x)*cos(2*pi*y) \
                - 1.0*pi*sin(2*pi*y)*cos(2*pi*x) + (80*pi**2+2)*cos(2*pi*x)*cos(2*pi*y)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return bc_value(p, self.bcs, 'dirichlet')

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return bc_mask(p, self.bcs, 'dirichlet')

    def neumann(self, p: TensorLike) -> TensorLike:
        return bc_value(p, self.bcs, 'neumann')

    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        return bc_mask(p, self.bcs, 'neumann')
