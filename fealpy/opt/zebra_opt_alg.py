
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Zebra Optimization Algorithm
~~~~~~~~~~
Reference:
Eva Trojovska, Mohammad Dehghani, Pavel Trojovsky.
Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm for Solving Optimization Algorithm.
IEEE Access, 10: 49445-49473.
"""
class ZebraOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        
        for it in range(0, self.MaxIT):
            
            # exploration percentage and exploitation percentage
            self.D_pl_pt(it)

            # Foraging behavior
            x_new = self.x + bm.random.rand(self.N, self.dim) * (self.gbest - (1 + bm.random.rand(self.N, self.dim)) * self.x) # Eq.(3)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            r = bm.random.rand(self.N, 1)
            x_new = ((r < 0.5) * (self.x + 0.01 * self.x * (2 * bm.random.rand(self.N, self.dim) - 1) * (1 - it / self.MaxIT)) + # Against # Eq.(5).S1
                     (r >= 0.5) * (self.x + bm.random.rand(self.N, self.dim) * (self.x[bm.random.randint(0, self.N, (self.N,))] - (1 + bm.random.rand(self.N, self.dim)) * self.x))) # Attact # Eq.(5).S2
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f
