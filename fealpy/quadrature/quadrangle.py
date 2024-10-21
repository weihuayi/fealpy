
from ..backend import backend_manager as bm
from .gauss_legendre import GaussLegendreQuadrature


class QuadrangleQuadrature(GaussLegendreQuadrature):
    def make(self, index: int):
        bcs, ws = super().make(index)
        weights = bm.tensordot(ws, ws, axes=0)
        return (bcs, bcs), weights
