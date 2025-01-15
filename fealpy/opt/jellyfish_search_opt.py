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

        fit = self.fun(self.x)

        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        for it in range(self.MaxIT):
            self.D_pl_pt(it)

            c = (1 - it / self.MaxIT) * (2 * bm.random.rand(1) - 1)
            
            if c >= 0.5:
                x_new = self.x + bm.random.rand(self.N, 1) * (self.gbest - 3 * bm.random.rand(self.N, 1) * bm.mean(self.x, axis=0))
            else:
                r = bm.random.rand(self.N, 1)

                rand_index = bm.random.randint(0, self.N, (self.N,))
                Direction = ((fit[rand_index]) <= fit)[:, None] * (self.x[rand_index] - self.x) + ((fit[rand_index]) > fit)[:, None] * (self.x - self.x[rand_index])
                x_new = ((r > (1 - c)) * (self.x + 0.1 * bm.random.rand(self.N, self.dim) * (self.ub - self.lb)) + 
                        (r <= (1 - c)) * (self.x + bm.random.rand(self.N, self.dim) * Direction))
            
            x_new = x_new + ((self.ub + x_new - self.lb) - x_new) * (x_new < self.lb) + ((x_new - self.ub + self.lb) - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f