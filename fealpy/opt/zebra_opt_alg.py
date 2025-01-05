
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

        D_pl = bm.zeros((1, MaxIT))
        D_pt = bm.zeros((1, MaxIT))
        Div = bm.zeros((1, MaxIT))
        curve = bm.zeros((1, MaxIT))
        
        for it in range(0, MaxIT):
            Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            D_pl[0, it] = 100 * Div[0, it] / bm.max(Div)
            D_pt[0, it] = 100 * bm.abs(Div[0, it] - bm.max(Div)) / bm.max(Div)

            # Foraging behavior
            x_new = x + bm.random.rand(N, dim) * (gbest - (1 + bm.random.rand(N, dim)) * x) # Eq.(3)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)

            r = bm.random.rand(N, 1)
            x_new = ((r < 0.5) * (x + 0.01 * x * (2 * bm.random.rand(N, dim) - 1) * (1 - it / MaxIT)) + # Against # Eq.(5).S1
                     (r >= 0.5) * (x + bm.random.rand(N, dim) * (x[bm.random.randint(0, N, (N,))] - (1 + bm.random.rand(N, dim)) * x))) # Attact # Eq.(5).S2
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)

            curve[0, it] = gbest_f[0]

        self.gbest = gbest
        self.gbest_f = gbest_f[0]
        self.curve = curve[0]
        self.D_pl = D_pl[0]
        self.D_pt = D_pt[0]
