from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .opt_function import levy
from .optimizer_base import Optimizer

"""
Cuckoo Search Optimization

"""
class CuckooSearchOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, alpha=0.01, Pa=0.25):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            # Levy
            x_new = self.x + alpha * levy(self.N, self.dim, 1.5) * (self.x - self.gbest)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit) 
            
            # Nest
            x_new = self.x + bm.random.rand(self.N, self.dim) * bm.where((bm.random.rand(self.N, self.dim) - Pa) < 0, 0, 1) * (self.x[bm.random.randint(0, self.N, (self.N,))] - self.x[bm.random.randint(0, self.N, (self.N,))])
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f
