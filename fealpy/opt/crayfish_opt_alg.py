
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
        MaxIT = option["MaxIters"]
        dim = option["ndim"]
        lb, ub = option["domain"]
        
        fit = self.fun(x)       
        gbest_idx = bm.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = x[gbest_idx] 
        fit_new = bm.zeros((N,))
        curve = bm.zeros((1, MaxIT))
        D_pl = bm.zeros((1, MaxIT))
        D_pt = bm.zeros((1, MaxIT))
        Div = bm.zeros((1, MaxIT))
        global_position = gbest
        global_fitness = gbest_f
        for it in range(0, MaxIT):
            Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            D_pl[0, it] = 100 * Div[0, it] / bm.max(Div)
            D_pt[0, it] = 100 * bm.abs(Div[0, it] - bm.max(Div)) / bm.max(Div)
            
            C = 2 - (it / MaxIT)
            temp = bm.random.rand(1) * 15 + 20
            xf = (gbest + global_position) / 2
            p = 0.2 * (1 / (bm.sqrt(bm.array(2 * bm.pi) * 3))) * bm.exp(bm.array(- (temp - 25) ** 2 / (2 * 3 ** 2)))
            rand = bm.random.rand(N, 1)
            rr = bm.random.rand(4, N, dim)

            z = bm.random.randint(0, N, (N,))

            gbest = gbest.reshape(1, dim)

            P = 3 * bm.random.rand(N) * fit / (gbest_f + 2.2204e-16)
          
            x_new = ((temp > 30) * ((rand < 0.5) * (x + C * bm.random.rand(N, dim) * (xf - x)) + 
                                    (rand > 0.5) * (x - x[z] + xf)) + 
                     (temp <= 30) * ((P[:, None] > 2) * (x + bm.cos(2 * bm.random.rand(N, dim) * bm.pi) * gbest * p - bm.sin(2 * bm.pi * bm.random.rand(N, dim) * gbest * p)) + 
                                     (P[:, None] <= 2) * ((x - gbest) * p + p * bm.random.rand(N, dim) * x)))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)          
            fit_new = self.fun(x_new)
            # for i in range(N):
            #     fit_new[i] = self.fun(x_new[i])
            mask = fit_new < fit
            x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
            newbest_id = bm.argmin(fit_new)
            (global_position, global_fitness) = (x_new[newbest_id], fit_new[newbest_id]) if fit_new[newbest_id] < global_fitness else (global_position, global_fitness)
            gbest_idx = bm.argmin(fit)
            if fit[gbest_idx] < gbest_f:
                gbest_f = fit[gbest_idx]
                gbest = x[gbest_idx].reshape(1, dim)
            gbest = gbest.flatten()
            curve[0, it] = gbest_f
        self.gbest = gbest
        self.gbest_f = gbest_f
        self.curve = curve[0]
        self.D_pl = D_pl[0]
        self.D_pt = D_pt[0]