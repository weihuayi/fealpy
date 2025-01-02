from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

"""
StarFish Optimization Algorithm

Reference:
~~~~~~~~~~
Changting Zhong, Gang Li, Zeng Meng, Haijiang Li, Ali Riza Yildiz, Seyedali Mirjalili.
Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers.
Neural Computing and Applications, 2024.
"""

class StarFishOptAlg(Optimizer):
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
        x_new = bm.zeros((N, dim))
        for it in range(MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[0, it], self.D_pt[0, it] = self.D_pl_pt(self.Div[0, it])

            tEO = (MaxIT - it) / MaxIT * bm.cos(bm.array(bm.pi * it / (2 * MaxIT)))
            if bm.random.rand(1) > 0.5:
                if dim > 5:
                    j = bm.random.randint(0, dim, (N, dim))
                    r1 = bm.random.rand(N, 1)
                    x_new[bm.arange(N)[:, None], j] = ((r1 > 0.5) * 
                                                       (x[bm.arange(N)[:, None], j] + (2 * r1 - 1) * bm.pi  * bm.cos(bm.array(bm.pi * it / (2 * MaxIT))) * 
                                                        ((self.gbest[None, :] * bm.ones((N, dim)))[bm.arange(N)[:, None], j] - x[bm.arange(N)[:, None], j])) + 
                                                       (r1 <= 0.5) * 
                                                       (x[bm.arange(N)[:, None], j] - (2 * r1 - 1) * bm.pi * bm.sin(bm.array(bm.pi * it / (2 * MaxIT))) * 
                                                        ((self.gbest[None, :] * bm.ones((N, dim)))[bm.arange(N)[:, None], j] - x[bm.arange(N)[:, None], j])))
                    x[bm.arange(N)[:, None], j] = (x_new[bm.arange(N)[:, None], j] + 
                                                  (x_new[bm.arange(N)[:, None], j] > ub) * (x[bm.arange(N)[:, None], j] - x_new[bm.arange(N)[:, None], j]) + 
                                                  (x_new[bm.arange(N)[:, None], j] < lb) * (x[bm.arange(N)[:, None], j] - x_new[bm.arange(N)[:, None], j]))
                else:
                    j =bm.random.randint(0, dim, (N, 1))
                    a = (x[bm.random.randint(0, N, (N,))[:, None], j] - x[bm.arange(N)[:, None], j])
                    x_new[bm.arange(N)[:, None], j] = (((MaxIT - it) / MaxIT) * (bm.cos(bm.array(bm.pi * it / (2 * MaxIT)))) * x[bm.arange(N)[:, None], j] + 
                                                       (2 * bm.random.rand(N, 1) - 1) * (x[bm.random.randint(0, N, (N, 1)), j] - x[bm.arange(N)[:, None], j]) + 
                                                       (2 * bm.random.rand(N, 1) - 1) * (x[bm.random.randint(0, N, (N, 1)), j] - x[bm.arange(N)[:, None], j]))
                    x[bm.arange(N)[:, None], j] = (x_new[bm.arange(N)[:, None], j] + 
                                                  (x_new[bm.arange(N)[:, None], j] > ub) * (x[bm.arange(N)[:, None], j] - x_new[bm.arange(N)[:, None], j]) + 
                                                  (x_new[bm.arange(N)[:, None], j] < lb) * (x[bm.arange(N)[:, None], j] - x_new[bm.arange(N)[:, None], j]))
            else:
                dm = self.gbest - x[bm.random.randint(0, N - 1, (5,))]
                x_new[0 : N - 1] = x[0 : N - 1] + bm.random.rand(N - 1, dim) * dm[bm.random.randint(0, 5, (N - 1,))] + bm.random.rand(N - 1, dim) * dm[bm.random.randint(0, 5, (N - 1,))]
                x_new[N - 1] = x[N - 1] * bm.exp(bm.array(-it * N / MaxIT))
                x = x_new + (x_new > ub) * (ub - x_new) + (x_new < lb) * (lb - x_new)
            fit = self.fun(x)[:, None]
            gbest_idx = bm.argmin(fit)
            (self.gbest, self.gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < self.gbest_f else (self.gbest, self.gbest_f)
            self.curve[0, it] = self.gbest_f[0]

        self.curve = self.curve.flatten()
        self.D_pl = self.D_pl.flatten()
        self.D_pt = self.D_pt.flatten()