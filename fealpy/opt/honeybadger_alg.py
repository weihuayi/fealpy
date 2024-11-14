
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
        T = option["MaxIters"]
        fit = self.fun(x)[:, None]
        dim = option["ndim"]
        lb, ub = option["domain"]
        gbest_idx = bm.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = x[gbest_idx] 
        C = 2
        eps = 2.2204e-16
        beta = 6
        for t in range (0, T):
            alpha = C * bm.exp(bm.array(-t/T))
            di = ((bm.linalg.norm(x - gbest +eps, axis=1)) ** 2)[:, None]
            S = ((bm.linalg.norm(x - bm.concatenate((x[1:], x[0:1])) + eps, 2, axis=1)) ** 2)[:, None]
            r2 = bm.random.rand(N, 1)
            I = r2 * S / (4 * bm.pi * di)
            F = bm.where(bm.random.rand(N, 1) < 0.5, bm.array(1), bm.array(-1))
            r3 = bm.random.rand(N, dim)
            r4 = bm.random.rand(N, dim)
            r5 = bm.random.rand(N, dim)
            r7 = bm.random.rand(N, dim) 
            r = bm.random.rand(N, 1)
            di = gbest - x
            x_new = (
                      (r < 0.5) * 
                      (gbest + F * beta * I * gbest + F * r3 * alpha * di * bm.abs(bm.cos(2 * bm.pi * r4) * (1 - bm.cos(2 * bm.pi * r5)))) + 
                      (r >= 0.5) * 
                      (gbest + F * r7 * alpha * di))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            # print("HBA: The optimum at iteration", t + 1, "is", gbest_f) 
        return gbest, gbest_f
