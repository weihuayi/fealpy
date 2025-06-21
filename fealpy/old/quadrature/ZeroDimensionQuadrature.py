import numpy as np
from .Quadrature import Quadrature


class ZeroDimensionQuadrature(Quadrature):

    def __init__(self, index):
        self.quadpts = np.array([[1]], dtype=np.float64)
        self.weights = np.array([1], dtype=np.float64)

