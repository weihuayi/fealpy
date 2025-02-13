
from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer


"""
Honey Badger Algorithm

Reference
~~~~~~~~~
Fatma A. Hashim, Essam H. Houssein, Kashif Hussain, Mai S. Mabrouk, Walid Al-Atabany.
Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems.
Mathematics and Computers in Simulation, 2022, 192: 84-110.
"""

class HoneybadgerAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    
    def run(self, C=2, beta=6):
        fit = self.fun(self.x)

        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx] 

        eps = 2.2204e-16

        for it in range (0, self.MaxIT):
            self.D_pl_pt(it)

            alpha = C * bm.exp(bm.array(-it / self.MaxIT))
            di = ((bm.linalg.norm(self.x - self.gbest +eps, axis=1)) ** 2)[:, None]
            S = ((bm.linalg.norm(self.x - bm.concatenate((self.x[1:], self.x[0:1])) + eps, 2, axis=1)) ** 2)[:, None]
            r2 = bm.random.rand(self.N, 1)
            I = r2 * S / (4 * bm.pi * di)
            F = bm.where(bm.random.rand(self.N, 1) < 0.5, bm.array(1), bm.array(-1))
            r = bm.random.rand(self.N, 1)
            di = self.gbest - self.x
            x_new = ((r < 0.5) * 
                    (self.gbest + F * beta * I * self.gbest + F * bm.random.rand(self.N, self.dim) * alpha * di * bm.abs(bm.cos(2 * bm.pi * bm.random.rand(self.N, self.dim)) * (1 - bm.cos(2 * bm.pi * bm.random.rand(self.N, self.dim))))) + 
                    (r >= 0.5) * 
                    (self.gbest + F * bm.random.rand(self.N, self.dim)  * alpha * di))
                      
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:,None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f