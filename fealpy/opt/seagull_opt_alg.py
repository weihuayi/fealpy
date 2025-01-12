
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Seagull Optimization Algorithm
~~~~~~~~~~
Reference:
Gaurav Dhiman, Vijay Kumar.
Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems.
Knowledge-Based Systems, 2019, 165: 169-196.
"""
class SeagullOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, Fc=2, u=1, v=1):
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
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))

        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            x = (bm.abs((Fc - (it * Fc / MaxIT)) * x + 2 * ((Fc - (it * Fc / MaxIT)) ** 2) * bm.random.rand(N, 1) * (self.gbest - x)) * 
                 u * bm.exp(v * bm.random.rand(N, 1) * 2 * bm.pi) * bm.cos(bm.random.rand(N, 1) * 2 * bm.pi) * 
                 u * bm.exp(v * bm.random.rand(N, 1) * 2 * bm.pi) * bm.sin(bm.random.rand(N, 1) * 2 * bm.pi) * 
                 u * bm.exp(v * bm.random.rand(N, 1) * 2 * bm.pi) * bm.random.rand(N, 1) * 2 * bm.pi + 
                 self.gbest) # Eq.(14)
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f