
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
import random
from .optimizer_base import Optimizer, opt_alg_options

"""
Crayfish Optimization Algorithm

Reference
~~~~~~~~~
Heming Jia, Honghua Rao, Changsheng Wen, Seyedali Mirjalili. 
Crayfish optimization algorithm. 
Artificial Intelligence Review, 2023, 56: S1919-S1979.

"""

class CrayfishOptAlg(Optimizer):

    def __init__(self, option) -> None:
        super().__init__(option)
    
    
    def run(self):

        option = self.options
        x = option["x0"]
        N =  option["NP"]
        T = option["MaxIters"]
        dim = option["ndim"]
        lb, ub = option["domain"]
        
        fit = self.fun(x)       
        gbest_idx = bm.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = x[gbest_idx] 
       

        global_position = gbest
        global_fitness = gbest_f
        for t in range(0, T):
            C = 2 - (t / T)
            temp = bm.random.rand(1) * 15 + 20
            xf = (gbest + global_position) / 2
            p = 0.2 * (1 / (bm.sqrt(bm.array(2 * bm.pi) * 3))) * bm.exp(bm.array(- (temp - 25) ** 2 / (2 * 3 ** 2)))
            rand = bm.random.rand(N, 1)
            rr = bm.random.rand(4, N, dim)
            # z = [random.randint(0, N - 1) for _ in range(N)]
            z = bm.random.randint(0, N - 1, (N,))

            gbest = gbest.reshape(1, dim)

            P = 3 * bm.random.rand(N) * fit / (gbest_f + 2.2204e-16)
          
            x_new = ((temp > 30) * ((rand < 0.5) * (x + C * rr[0] * (xf -x)) + 
                                    (rand > 0.5) * (x - x[z] + xf)) + 
                     (temp <= 30) * ((P[:, None] > 2) * (x + bm.cos(2 * rr[1] * bm.pi) * gbest * p - bm.sin(2 * bm.pi * rr[2] * gbest * p)) + 
                                     (P[:, None] <= 2) * ((x - gbest) * p + p * rr[3] * x)))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)          
            fit_new = self.fun(x_new)

            mask = fit_new < fit
            x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
            newbest_id = bm.argmin(fit_new)
            (global_position, global_fitness) = (x_new[newbest_id], fit_new[newbest_id]) if fit_new[newbest_id] < global_fitness else (global_position, global_fitness)
            gbest_idx = bm.argmin(fit)
            if fit[gbest_idx] < gbest_f:
                gbest_f = fit[gbest_idx]
                gbest = x[gbest_idx].reshape(1, dim)
            gbest = gbest.flatten()
        return gbest, gbest_f
