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


    def run(self):
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)[:, None]
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

        # Parameters
        p = 0.9

        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            

            R = bm.random.rand(N, 1)
            
            # Attacking behavior
            n = 0.05 * bm.exp(bm.array(-2 * ((it / MaxIT) ** 2))) # Eq.(6)
            x_new = ((p < R) * (x + n * (1 + bm.sin(R)) * x) + 
                     (p >= R) * (x * (n * (2 * bm.random.rand(N, dim) - 1) + 1))) # Eq.(5)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            
            # Migration behavior
            m = 2 * bm.sin(R + bm.pi / 2) # Eq.(8)
            s = bm.random.randint(0, int(0.3 * N), (N,))
            fit_r = fit[s]
            cauchy_num = 1 / (bm.pi * ((bm.random.rand(N, dim) * bm.pi - bm.pi / 2) ** 2 + 1))
            x_new = ((fit < fit_r) * (x + cauchy_num * (x - self.gbest)) + 
                     (fit >= fit_r) * (x + cauchy_num * (self.gbest - m * x))) # Eq.(7)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f