from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer

"""
Artificial Rabbits Optimization

Reference:
~~~~~~~~~~
Liying Wang, Qingjiao Cao, Zhenxing Zhang, Seyedali Mirjalili, Weiguo Zhao.
Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems.
Engineering Applications of Artificial Intelligence, 2022, 114: 105082.
"""

class ArtificialRabbitsOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)
        
    def run(self):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        for it in range(self.MaxIT):
            self.D_pl_pt(it)

            A = 4 * (1 - it / self.MaxIT) * bm.log(1 / bm.random.rand(1))
            R = (bm.exp(bm.array(1)) - bm.exp(bm.array(((it - 1) / self.MaxIT) ** 2))) * bm.sin(2 * bm.pi * bm.random.rand(self.N, 1)) * bm.random.randint(0, 2, (self.N, self.dim))

            rand_index = bm.random.randint(0, self.N, (self.N,))

            r4 = bm.random.rand(self.N, 1)
            H = (self.MaxIT - it + 1) * r4 / self.MaxIT
            k = bm.random.randint(0, self.dim, (self.N,))
            g = bm.zeros((self.N, self.dim))
            g[bm.arange(self.N), k] = 1
            b = self.x + H * g * self.x
            if A > 1:
                x_new = self.x[rand_index] + R * (self.x - self.x[rand_index]) + bm.round(0.5 * (0.05 + bm.random.rand(self.N, self.dim))) * bm.random.randn(self.N, self.dim)
            else:
                x_new = self.x + R * (r4 * b - self.x)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f