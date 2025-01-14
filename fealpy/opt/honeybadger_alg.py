
from fealpy.backend import backend_manager as bm 
from fealpy.typing import TensorLike, Index, _S
from fealpy import logger
from fealpy.opt.optimizer_base import Optimizer


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


    
    def run(self):

        option = self.options
        x = option["x0"]
        N =  option["NP"]
        MaxIT = option["MaxIters"]
        fit = self.fun(x)
        dim = option["ndim"]
        lb, ub = option["domain"]
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = x[gbest_idx] 
        C = 2
        eps = 2.2204e-16
        beta = 6
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        for it in range (0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            alpha = C * bm.exp(bm.array(-it / MaxIT))
            di = ((bm.linalg.norm(x - self.gbest +eps, axis=1)) ** 2)[:, None]
            S = ((bm.linalg.norm(x - bm.concatenate((x[1:], x[0:1])) + eps, 2, axis=1)) ** 2)[:, None]
            r2 = bm.random.rand(N, 1)
            I = r2 * S / (4 * bm.pi * di)
            F = bm.where(bm.random.rand(N, 1) < 0.5, bm.array(1), bm.array(-1))
            r = bm.random.rand(N, 1)
            di = self.gbest - x
            x_new = ((r < 0.5) * 
                    (self.gbest + F * beta * I * self.gbest + F * bm.random.rand(N, dim) * alpha * di * bm.abs(bm.cos(2 * bm.pi * bm.random.rand(N, dim)) * (1 - bm.cos(2 * bm.pi * bm.random.rand(N, dim))))) + 
                    (r >= 0.5) * 
                    (self.gbest + F * bm.random.rand(N, dim)  * alpha * di))
                      
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            x, fit = bm.where(mask[:,None], x_new, x), bm.where(mask, fit_new, fit)
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f
        # self.gbest = gbest
        # self.gbest_f = gbest_f[0]
        # self.curve = curve[0]
        # self.D_pl = D_pl[0]
        # self.D_pt = D_pt[0]
