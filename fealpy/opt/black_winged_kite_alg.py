from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Black_winged Kite Algorithm

Reference
~~~~~~~~~
Jun Wang, Wen-chuan Wang, Xiao-xue Hu, Lin Qiu, Hong-fei Zang.
Black-winged kite algorithm: a nature-inspired meta-heuristic for solving benchmark functions and engineering problems.
Artificial Intelligence Review, 2024, 2024: 57-98.
"""
class BlackwingedKiteAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, p=0.9):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            R = bm.random.rand(self.N, 1)
            
            # Attacking behavior
            n = 0.05 * bm.exp(bm.array(-2 * ((it / self.MaxIT) ** 2))) # Eq.(6)
            x_new = ((p < R) * (self.x + n * (1 + bm.sin(R)) * self.x) + 
                     (p >= R) * (self.x * (n * (2 * bm.random.rand(self.N, self.dim) - 1) + 1))) # Eq.(5)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Migration behavior
            m = 2 * bm.sin(R + bm.pi / 2) # Eq.(8)
            s = bm.random.randint(0, int(0.3 * self.N), (self.N,))
            fit_r = fit[s]
            cauchy_num = 1 / (bm.pi * ((bm.random.rand(self.N, self.dim) * bm.pi - bm.pi / 2) ** 2 + 1))
            x_new = ((fit < fit_r)[:, None] * (self.x + cauchy_num * (self.x - self.gbest)) + 
                     (fit >= fit_r)[:, None] * (self.x + cauchy_num * (self.gbest - m * self.x))) # Eq.(7)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f