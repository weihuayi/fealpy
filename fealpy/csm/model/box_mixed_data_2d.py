from typing import Optional, Sequence
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ..box_domain_mesher import BoxDomainMesher2d



class BoxMixedData2d(BoxDomainMesher2d):
    def __init__(self, box: Optional[Sequence[float]] = None):
        
        if box is None:
            box = [-1, 1, -1, 1]
        super().__init__(box)
        self.hypo = 'plane_strain'

    def lam(self, p: Optional[TensorLike] = None) -> float:
        return 1.0  

    def mu(self, p: Optional[TensorLike] = None) -> float:
        return 0.5  

    def stress_matrix_coefficient(self) -> tuple[float, float]:

        d = self.geo_dimension()
        lam, mu = self.lam(), self.mu()
        λ0 = 1.0 / (2 * mu)
        λ1 = lam / (2 * mu * (d * lam + 2 * mu))
        return λ0, λ1

    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        shp = list(p.shape[:-1]) + [2]
        return bm.ones(tuple(shp), dtype=p.dtype)

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        shp = list(p.shape[:-1]) + [2]
        return bm.zeros(tuple(shp), dtype=p.dtype)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[...,0], p[...,1]
        eps = 1e-12
        box = self.domain()  
        flag = (bm.abs(x-box[0])<eps) | (bm.abs(x-box[1])<eps) \
             | (bm.abs(y-box[2])<eps) | (bm.abs(y-box[3])<eps)
        return flag






