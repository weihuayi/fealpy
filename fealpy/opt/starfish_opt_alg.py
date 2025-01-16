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
        
        fit = self.fun(self.x)
        
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        
        x_new = bm.zeros((self.N, self.dim))
        for it in range(self.MaxIT):
            self.D_pl_pt(it)
            
            if bm.random.rand(1) > 0.5:
                if self.dim > 5:
                    j = bm.random.randint(0, self.dim, (self.N, self.dim))
                    r1 = bm.random.rand(self.N, 1)
                    x_new[bm.arange(self.N)[:, None], j] = ((r1 > 0.5) * 
                                                       (self.x[bm.arange(self.N)[:, None], j] + (2 * r1 - 1) * bm.pi  * bm.cos(bm.array(bm.pi * it / (2 * self.MaxIT))) * 
                                                        ((self.gbest[None, :] * bm.ones((self.N, self.dim)))[bm.arange(self.N)[:, None], j] - self.x[bm.arange(self.N)[:, None], j])) + 
                                                       (r1 <= 0.5) * 
                                                       (self.x[bm.arange(self.N)[:, None], j] - (2 * r1 - 1) * bm.pi * bm.sin(bm.array(bm.pi * it / (2 * self.MaxIT))) * 
                                                        ((self.gbest[None, :] * bm.ones((self.N, self.dim)))[bm.arange(self.N)[:, None], j] - self.x[bm.arange(self.N)[:, None], j])))
                    self.x[bm.arange(self.N)[:, None], j] = (x_new[bm.arange(self.N)[:, None], j] + 
                                                  (x_new[bm.arange(self.N)[:, None], j] > self.ub) * (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j]) + 
                                                  (x_new[bm.arange(self.N)[:, None], j] < self.lb) * (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j]))
                else:
                    j =bm.random.randint(0, self.dim, (self.N, 1))
                    a = (self.x[bm.random.randint(0, self.N, (self.N,))[:, None], j] - self.x[bm.arange(self.N)[:, None], j])
                    x_new[bm.arange(self.N)[:, None], j] = (((self.MaxIT - it) / self.MaxIT) * bm.cos(bm.array(bm.pi * it / (2 * self.MaxIT))) * self.x[bm.arange(self.N)[:, None], j] + 
                                                       (2 * bm.random.rand(self.N, 1) - 1) * (self.x[bm.random.randint(0, self.N, (self.N, 1)), j] - self.x[bm.arange(self.N)[:, None], j]) + 
                                                       (2 * bm.random.rand(self.N, 1) - 1) * (self.x[bm.random.randint(0, self.N, (self.N, 1)), j] - self.x[bm.arange(self.N)[:, None], j]))
                    self.x[bm.arange(self.N)[:, None], j] = (x_new[bm.arange(self.N)[:, None], j] + 
                                                  (x_new[bm.arange(self.N)[:, None], j] > self.ub) * (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j]) + 
                                                  (x_new[bm.arange(self.N)[:, None], j] < self.lb) * (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j]))
            else:
                dm = self.gbest - self.x[bm.random.randint(0, self.N - 1, (5,))]
                x_new[0 : self.N - 1] = self.x[0 : self.N - 1] + bm.random.rand(self.N - 1, self.dim) * dm[bm.random.randint(0, 5, (self.N - 1,))] + bm.random.rand(self.N - 1, self.dim) * dm[bm.random.randint(0, 5, (self.N - 1,))]
                x_new[self.N - 1] = self.x[self.N - 1] * bm.exp(bm.array(-it * self.N / self.MaxIT))
                self.x = x_new + (x_new > self.ub) * (self.ub - x_new) + (x_new < self.lb) * (self.lb - x_new)
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f