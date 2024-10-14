
import torch

from .gauss_legendre import GaussLegendreQuadrature


class QuadrangleQuadrature(GaussLegendreQuadrature):
    def make(self, index: int):
        bcs, ws = super().make(index)
        weights = torch.einsum('i, j->ij', ws, ws)
        return (bcs, bcs), weights
