import numpy as np

from .GaussLegendreQuadrture import GaussLegendreQuadrture 

class QuadrangleQuadrature():
    def __init__(self, k):
        qf = GaussLegendreQuadrture(k)
        bcs, ws = qf.quadpts, qf.weights
        self.quadpts = np.zeros((len(ws)**2, 4), dtype=np.float)
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

    def get_number_of_quad_points(self):
        return self.quadpts.shape[0] 

    def get_gauss_point_and_weight(self, i):
        return self.quadpts[i,:], self.weights[i] 

