import numpy as np
from .Quadrature import Quadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature


class QuadrangleQuadrature(Quadrature):
    def __init__(self, k):
        q0 = GaussLegendreQuadrature(k)
        nq = q0.number_of_quadrature_points()
        bcs, ws = q0.get_quadrature_points_and_weights()
        self.quadpts = np.zeros((nq**2, 4), dtype=np.float)
        X, Y = np.meshgrid(bcs[:, 0], bcs[:, 0])
        self.quadpts[:, 0] = (X*Y).flat
        X, Y = np.meshgrid(bcs[:, 1], bcs[:, 0])
        self.quadpts[:, 1] = (X*Y).flat
        X, Y = np.meshgrid(bcs[:, 1], bcs[:, 1])
        self.quadpts[:, 2] = (X*Y).flat
        X, Y = np.meshgrid(bcs[:, 0], bcs[:, 1])
        self.quadpts[:, 3] = (X*Y).flat
        W1, W2 = np.meshgrid(ws, ws)
        self.weights = (W1*W2).flatten()
