
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Butterfly Optimization Algorithm

"""
class ButterflyOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, a=0.1, c=0.01, p=0.8):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            f = c * (fit ** a) # Fragrance Eq.(1)

            rand = bm.random.rand(self.N, 1)
            x_new = ((rand < p) * (self.x + ((bm.random.rand(self.N, self.dim) ** 2) * self.gbest - self.x) * f[:, None]) + # Global search Eq.(2)
                     (rand >= p) * (self.x + ((bm.random.rand(self.N, self.dim) ** 2) * self.x[bm.random.randint(0, self.N, (self.N,))] - self.x[bm.random.randint(0, self.N, (self.N,))]) * f[:, None])) # Local search Eq.(3)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f
