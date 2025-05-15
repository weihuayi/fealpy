from typing import Callable, Dict, Literal, Sequence

from ...backend import TensorLike 
from ...backend import backend_manager as bm

from ..boundary_condition import BoundaryCondition, bc_mask, bc_value



class SinSinData2D():
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
        """
        Diffusion coefficient
        """
        val = bm.array([[1.0, 0.0], [0.0, 1.0]], **bm.context(p))
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    def reaction_coef(self, p: TensorLike) -> TensorLike:
        """
        Reaction coefficient
        """
        val = bm.array([1.0], **bm.cotext(p))
        shape = p.shape[:-1] + val.shape
        return bm.broadcast_to(val, shape)

    def solution(self, p: TensorLike) -> TensorLike:
        """
        Analytical solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.sin(pi*x)*bm.sin(pi*y)
        return val # val.shape == x.shape

    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Gradient of the solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.stack((
            pi*bm.cos(pi*x)*bm.sin(pi*y),
            pi*bm.sin(pi*x)*bm.cos(pi*y)), axis=-1)
        return val # val.shape == p.shape

    def flux(self, p: TensorLike) -> TensorLike:
        """
        Flux of the solution
        """
        grad = self.gradient(p)
        val = self.diffusion_coef(p) 
        val = bm.einsum('...ij, ...j->...i', val, -grad)
        return val

    def source(self, p: TensorLike) -> TensorLike:
        """
        Source term
        """
        pi = bm.pi
        val = (2*pi**2 + 1.0)*self.solution(p)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return bc_value(p, self.bcs, 'dirichlet')

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return bc_mask(p, self.bcs, 'dirichlet')

    def neumann(self, p: TensorLike) -> TensorLike:
        return bc_value(p, self.bcs, 'neumann')

    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        return bc_mask(p, self.bcs, 'neumann')
