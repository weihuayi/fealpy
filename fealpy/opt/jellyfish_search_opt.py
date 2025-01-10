from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

"""
Jellyfish Search Optimization  

Reference:
~~~~~~~~~~
Jui-Sheng Chou, Dinh-Nhat Truong.
A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean.
Applied Mathematics and Computation, 2021, 389: 125535.
"""

class JellyfishSearchOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)
        
    def run(self):
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
        for it in range(MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            c = (1 - it / MaxIT) * (2 * bm.random.rand(1) - 1)
            
            if c >= 0.5:
                x_new = x + bm.random.rand(N, 1) * (self.gbest - 3 * bm.random.rand(N, 1) * bm.mean(x, axis=0))
            else:
                r = bm.random.rand(N, 1)

                rand_index = bm.random.randint(0, N, (N,))
                Direction = ((fit[rand_index]) <= fit)[:, None] * (x[rand_index] - x) + ((fit[rand_index]) > fit)[:, None] * (x - x[rand_index])
                x_new = ((r > (1 - c)) * (x + 0.1 * bm.random.rand(N, dim) * (ub - lb)) + 
                        (r <= (1 - c)) * (x + bm.random.rand(N, dim) * Direction))
            
            x_new = x_new + ((ub + x_new - lb) - x_new) * (x_new < lb) + ((x_new - ub + lb) - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f