import numpy as np
from scipy.special import j_roots, factorial

class StroudQuadrature:
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        p, self.weights = self._compute_quadrature()
        self.points = self._to_simplex(p)

    def _to_simplex(self, points):
        d = self.dim
        shape = points.shape[:-1]
        bcs = np.zeros(shape+(d+1, ), dtype=np.float64)
        bcs[:, 0] = points[:, 0]
        for i in range(1, d):
            bcs[:, i] = points[:, i] * (1-bcs[:, :i].sum(axis=-1))
        bcs[:, d] = 1-bcs[:, :d].sum(axis=-1)
        return bcs

    def _compute_quadrature(self):
        d = self.dim
        n = self.n

        points = []
        weights = []
        for i in range(1, d+1):
            p, w, s = j_roots(n, d-i, 0, mu=True)
            points.append((p+1)/2)
            weights.append(w/s)
        points = np.meshgrid(*points)
        weights = np.meshgrid(*weights)

        points = np.array([p.flatten() for p in points]).T
        weights = np.prod([w.flatten() for w in weights], axis=0)#/factorial(d)
        return points, weights

    def get_points(self):
        return self.points

    def get_weights(self):
        return self.weights

    def get_points_and_weights(self):
        return self.points, self.weights

    get_quadrature_points_and_weights = get_points_and_weights














