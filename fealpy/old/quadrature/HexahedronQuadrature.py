import numpy as np
from .Quadrature import Quadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature


class HexahedronQuadrature(Quadrature):
    def __init__(self, index, dtype=np.float64):
        q0 = GaussLegendreQuadrature(index)
        bcs, ws = q0.get_quadrature_points_and_weights()
        self.quadpts = (bcs, bcs, bcs)
        self.weights = np.einsum('i, j, k->ijk', ws, ws, ws)

    def number_of_quadrature_points(self):
        n = self.quadpts[0].shape[0]
        return n*n*n
