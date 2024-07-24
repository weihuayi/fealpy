
from .quadrature import fealpy
from .gauss_legendre import GaussLegendreQuadrature


class QuadrangleQuadrature(GaussLegendreQuadrature):
    def make(self, index: int):
        bcs, ws = super().make(index)
        weights = fealpy.einsum('i, j->ij', ws, ws)
        return (bcs, bcs), weights
