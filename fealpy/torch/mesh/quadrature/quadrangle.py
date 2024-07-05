import torch
from .quadrature import Quadrature
from .gauss_legendre import GaussLegendreQuadrature


class QuadrangleQuadrature(Quadrature):
    def make(self, index: int):
        #kwargs = {'dtype': self.dtype, 'device': self.device}
        q0 = GaussLegendreQuadrature(index)
        bcs, ws = q0.get_quadrature_points_and_weights()
        self.quadpts = (bcs, bcs)
        self.weights = torch.einsum('i, j->ij', ws, ws)