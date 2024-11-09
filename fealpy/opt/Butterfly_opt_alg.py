
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

        # Parameters
        c = 0.01
        p = 0.8
        a = 0.1

        for it in range(0, MaxIT):
            
            f = c * (fit ** a) # Fragrance Eq.(1)

            rand = bm.random.rand(N, 1)
            x_new = ((rand < p) * (x + ((bm.random.rand(N, 1) ** 2) * gbest - x) * f) + # Global search Eq.(2)
                     (rand >= p) * (x + ((bm.random.rand(N, 1) ** 2) * x[bm.random.randint(0, N, (N,))] - x[bm.random.randint(0, N, (N,))]) * f)) # Local search Eq.(3)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
        return gbest, gbest_f
