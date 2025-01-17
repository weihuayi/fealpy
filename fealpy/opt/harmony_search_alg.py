
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
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Parametes
        FW = 0.02 * (self.ub - self.lb)
        nNew = int(0.8 * self.N)
        HMCR = 0.9
        PAR = 0.1
        FW_damp = 0.995

        index = bm.argsort(fit)
        self.x = self.x[index]
        fit = fit[index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            x_new = initialize(nNew, self.dim, self.lb, self.ub)
            mask = bm.random.rand(nNew, self.dim) <= HMCR
            b = self.x[bm.random.randint(0, self.N, (nNew, self.dim)), bm.arange(self.dim)]
            x_new = mask * b + ~mask * x_new

            x_new = x_new + FW * bm.random.randn(nNew, self.dim) * (bm.random.rand(nNew, self.dim) <= PAR)

            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)

            self.x = bm.concatenate((self.x, x_new), axis=0)
            fit = bm.concatenate((fit, fit_new))

            index = bm.argsort(fit)
            self.x = self.x[index[0 : self.N]]
            fit = fit[index[0 : self.N]]
            self.update_gbest(self.x, fit)
            FW = FW * FW_damp
            self.curve[it] = self.gbest_f