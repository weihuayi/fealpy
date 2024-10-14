import numpy as np
from .Quadrature import Quadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature
from .TriangleQuadrature import TriangleQuadrature


class PrismQuadrature(Quadrature):
    def __init__(self, index, dtype=np.float64):
        q0 = GaussLegendreQuadrature(index)
        q1 = TriangleQuadrature(index)
        bc0, ws0 = q0.get_quadrature_points_and_weights()
        bc1, ws1 = q1.get_quadrature_points_and_weights()
        self.quadpts = (bc0, bc1)
        self.weights = np.einsum('i, j->ij', ws0, ws1)

    def number_of_quadrature_points(self):
        n0 = self.quadpts[0].shape[0]
        n1 = self.quadpts[1].shape[0]
        return n0*n1
