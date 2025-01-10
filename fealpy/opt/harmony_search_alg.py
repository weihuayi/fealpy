
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
        fit = self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest = x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Parametes
        FW = 0.02 * (ub - lb)
        nNew = int(0.8 * N)
        HMCR = 0.9
        PAR = 0.1
        FW_damp = 0.995

        index = bm.argsort(fit)
        x = x[index]
        fit = fit[index]


        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))

        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            
            x_new = initialize(nNew, dim, lb, ub)
            mask = bm.random.rand(nNew, dim) <= HMCR
            b = x[bm.random.randint(0, N, (nNew, dim)), bm.arange(dim)]
            x_new = mask * b + ~mask * x_new

            x_new = x_new + FW * bm.random.randn(nNew, dim) * (bm.random.rand(nNew, dim) <= PAR)

            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)

            x = bm.concatenate((x, x_new), axis=0)
            fit = bm.concatenate((fit, fit_new))

            index = bm.argsort(fit)
            x = x[index[0 : N]]
            fit = fit[index[0 : N]]
            self.update_gbest(x, fit)
            FW = FW * FW_damp
            self.curve[it] = self.gbest_f