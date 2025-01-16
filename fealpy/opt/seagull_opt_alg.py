
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
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            self.x = (bm.abs((Fc - (it * Fc / self.MaxIT)) * self.x + 2 * ((Fc - (it * Fc / self.MaxIT)) ** 2) * bm.random.rand(self.N, 1) * (self.gbest - self.x)) * 
                      u * bm.exp(v * bm.random.rand(self.N, 1) * 2 * bm.pi) * bm.cos(bm.random.rand(self.N, 1) * 2 * bm.pi) * 
                      u * bm.exp(v * bm.random.rand(self.N, 1) * 2 * bm.pi) * bm.sin(bm.random.rand(self.N, 1) * 2 * bm.pi) * 
                      u * bm.exp(v * bm.random.rand(self.N, 1) * 2 * bm.pi) * bm.random.rand(self.N, 1) * 2 * bm.pi + 
                      self.gbest) # Eq.(14)
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f