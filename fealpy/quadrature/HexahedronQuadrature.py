import numpy as np
from .Quadrature import Quadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature


class HexahedronQuadrature(Quadrature):
    def __init__(self, index, dtype=np.float):
        q0 = GaussLegendreQuadrature(index)
        nq = q0.number_of_quadrature_points()
        bcs, ws = q0.get_quadrature_points_and_weights()
        self.quadpts = np.zeros((nq**3, 8), dtype=dtype)
        i0, i1, i2 = np.meshgrid([0, 1], [0, 1], [0, 1])
        idx = np.zeros((8, 3), dtype=np.int8)
        idx[:, 0] = i0.flat
        idx[:, 1] = i1.flat
        idx[:, 2] = i2.flat
        idx = idx[[0, 2, 6, 4, 1, 3, 7, 5]]
        print(idx)
        for i in range(8):
            X, Y, Z = np.meshgrid(
                    bcs[:, idx[i, 0]],
                    bcs[:, idx[i, 1]],
                    bcs[:, idx[i, 2]])
            self.quadpts[:, i] = (X*Y*Z).flat

        w0, w1, w2 = np.meshgrid(ws, ws, ws)
        self.weights = (w0*w0*w0).flatten()
