import numpy as np
from .Quadrature import Quadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature


class QuadrangleQuadrature(Quadrature):
    def __init__(self, k):
        q0 = GaussLegendreQuadrature(k)
        bcs, ws = q0.get_quadrature_points_and_weights()
        self.quadpts = (bcs, bcs)
        self.weights = np.einsum('i, j->ij', ws, ws)

    def number_of_quadrature_points(self):
        n = self.quadpts[0].shape[0]
        return n*n
