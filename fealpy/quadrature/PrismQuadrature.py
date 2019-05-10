import numpy as np
from .Quadrature import Quadrature
from .IntervalQuadrature import IntervalQuadrature
from .TriangleQuadrature import TriangleQuadrature


class PrismQuadrature(Quadrature):
    def __init__(self, index, dtype=np.float):
        q0 = IntervalQuadrature(index)
        q1 = TriangleQuadrature(index)
        n0 = q0.number_of_quadrature_points()
        n1 = q1.number_of_quadrature_points()
        bc0, ws0 = q0.get_quadrature_points_and_weights()
        bc1, ws1 = q1.get_quadrature_points_and_weights()
        self.quadpts = np.zeros((n0*n1, 5), dtype=dtype)
        self.weights = np.zeros(n0*n1, dtype=dtype)
        self.quadpts[:, 0:2] = np.repeat(bc0, n1, axis=0)
        self.quadpts[:, 2:] = np.tile(bc1, (n0, 1))
        self.weights = np.tile(ws1, n0)*np.repeat(ws0, n1)
