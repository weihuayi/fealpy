from typing import Optional

from ...backend import bm 
from ...typing import TensorLike
from ...decorator import cartesian
from ..box_domain_mesher import BoxDomainMesher3d


class BoxDomainData3d(BoxDomainMesher3d):

    def __init__(self):
        super().__init__(box=[0, 1, 0, 0.2, 0, 0.2])

        self.L = 1.0 # length of the box in x direction
        self.W = 0.2 # width of the box in y and z direction

        delta = self.W/self.L # aspect ratio 
        self.g = 0.4 * delta**2 # gravity acceleration 
        self.d = bm.array(
                [0.0, 0.0, 1.0], 
                dtype=bm.float64,
                device=bm.get_device())
    
    @property
    def lam(self, p: Optional[TensorLike] = None) -> TensorLike:
        return 1.25 
    @property
    def mu(self, p: Optional[TensorLike] = None) -> TensorLike:
        return 1.0 
    @property
    def rho(self, p: Optional[TensorLike] = None) -> TensorLike:
        return 1.0 

    @cartesian
    def body_force(self, p: TensorLike):
        val = bm.zeros_like(p, **bm.context(p))
        return val

    @cartesian
    def displacement(self, p: TensorLike):
        raise NotImplementedError(
                "Displacement computation is not implemented for BoxDomainData3d.")

    @cartesian
    def strain(self, p: TensorLike) -> TensorLike:
        raise NotImplementedError(
                "Strain computation is not implemented for BoxDomainData3d.")

    @cartesian
    def stress(self, p: TensorLike) -> TensorLike:
        raise NotImplementedError(
                "Stress computation is not implemented for BoxDomainData3d.")

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        return bm.zeros_like(p, **bm.context(p))
    
    @cartesian
    def is_displacement_boundary(self, p: TensorLike) -> TensorLike:
        return bm.abs(p[..., 0]) < 1e-12 
