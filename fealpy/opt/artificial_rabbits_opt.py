from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

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


        self.curve = bm.zeros((1, MaxIT))
        self.D_pl = bm.zeros((1, MaxIT))
        self.D_pt = bm.zeros((1, MaxIT))
        self.Div = bm.zeros((1, MaxIT))
        for it in range(MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[0, it], self.D_pt[0, it] = self.D_pl_pt(self.Div[0, it])

            A = 4 * (1 - it / MaxIT) * bm.log(1 / bm.random.rand(1))
            R = (bm.exp(bm.array(1)) - bm.exp(bm.array(((it - 1) / MaxIT) ** 2))) * bm.sin(2 * bm.pi * bm.random.rand(N, 1)) * bm.random.randint(0, 2, (N, dim))
            # c = bm.random.randint(0, 2, (N, dim))
            rand_index = bm.random.randint(0, N, (N,))
            # g = bm.random.randint(0, dim, (N, dim))
            # l = bm.round(bm.random.rand(N, 1) * dim)
            r4 = bm.random.rand(N, 1)
            H = (MaxIT - it + 1) * r4 / MaxIT
            k = bm.random.randint(0, dim, (N,))
            g = bm.zeros((N, dim))
            g[bm.arange(N), k] = 1
            b = x + H * g * x
            if A > 1:
                x_new = x[rand_index] + R * (x - x[rand_index]) + bm.round(0.5 * (0.05 + bm.random.rand(N, dim))) * bm.random.randn(N, dim)
            else:
                x_new = x + R * (r4 * b - x)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (self.gbest, self.gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < self.gbest_f else (self.gbest, self.gbest_f)
            self.curve[0, it] = self.gbest_f[0]

        self.curve = self.curve.flatten()
        self.D_pl = self.D_pl.flatten()
        self.D_pt = self.D_pt.flatten()