
from .opt_function import levy
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .opt_function import initialize
from .optimizer_base import Optimizer

"""
Harmony Search Algorithm  

Reference:
~~~~~~~~~~
Zong Woo Geem, Joong Hoon Kim, G.V. Loganathan.
A New Heuristic Optimization Algorithm: Harmony Search.
Simulation, 2001, 76: 60-68.
"""

class HarmonySearchAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self):
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        gbest = x[gbest_index]
        gbest_f = fit[gbest_index]

        # Parametes
        FW = 0.02 * (ub - lb)
        nNew = int(0.8 * N)
        HMCR = 0.9
        PAR = 0.1
        FW_damp = 0.995

        index = bm.argsort(fit, axis=0)
        x = x[index[:, 0]]
        fit = fit[index[:, 0]]


        curve = bm.zeros((1, MaxIT))
        D_pl = bm.zeros((1, MaxIT))
        D_pt = bm.zeros((1, MaxIT))
        Div = bm.zeros((1, MaxIT))

        for it in range(0, MaxIT):
            Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            D_pl[0, it] = 100 * Div[0, it] / bm.max(Div)
            D_pt[0, it] = 100 * bm.abs(Div[0, it] - bm.max(Div)) / bm.max(Div)
            
            x_new = initialize(nNew, dim, lb, ub)
            mask = bm.random.rand(nNew, dim) <= HMCR
            b = x[bm.random.randint(0, N, (nNew, dim)), bm.arange(dim)]
            x_new = mask * b + ~mask * x_new

            x_new = x_new + FW * bm.random.randn(nNew, dim) * (bm.random.rand(nNew, dim) <= PAR)

            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]

            x = bm.concatenate((x, x_new), axis=0)
            fit = bm.concatenate((fit, fit_new), axis=0)

            index = bm.argsort(fit, axis=0)
            x = x[index[0 : N, 0]]
            fit = fit[index[0 : N, 0]]
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            FW = FW * FW_damp
            curve[0, it] = gbest_f

        self.gbest = gbest.flatten()
        self.gbest_f = gbest_f
        self.curve = curve.flatten()
        self.D_pl = D_pl[0]
        self.D_pt = D_pt[0]