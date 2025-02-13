
from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger

from ..optimizer_base import Optimizer, opt_alg_options

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
        
        fit = self.fun(self.x)       
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx] 
        fit_new = bm.zeros((self.N,))
        global_position = self.gbest
        global_fitness = self.gbest_f
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            C = 2 - (it / self.MaxIT)
            temp = bm.random.rand(1) * 15 + 20
            xf = (self.gbest + global_position) / 2
            p = 0.2 * (1 / (bm.sqrt(bm.array(2 * bm.pi) * 3))) * bm.exp(bm.array(- (temp - 25) ** 2 / (2 * 3 ** 2)))
            rand = bm.random.rand(self.N, 1)

            self.gbest = self.gbest.reshape(1, self.dim)

            P = 3 * bm.random.rand(self.N) * fit / (self.gbest_f + 2.2204e-16)
          
            x_new = ((temp > 30) * ((rand < 0.5) * (self.x + C * bm.random.rand(self.N, self.dim) * (xf - self.x)) + 
                                    (rand > 0.5) * (self.x - self.x[bm.random.randint(0, self.N, (self.N,))] + xf)) + 
                    (temp <= 30) * ((P[:, None] > 2) * (self.x + bm.cos(2 * bm.random.rand(self.N, self.dim) * bm.pi) * self.gbest * p - bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim) * self.gbest * p)) + 
                                   (P[:, None] <= 2) * ((self.x - self.gbest) * p + p * bm.random.rand(self.N, self.dim) * self.x)))
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)          
            fit_new = self.fun(x_new)
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            newbest_id = bm.argmin(fit_new)
            (global_position, global_fitness) = (x_new[newbest_id], fit_new[newbest_id]) if fit_new[newbest_id] < global_fitness else (global_position, global_fitness)
            self.update_gbest(self.x, fit)

            self.gbest = self.gbest.flatten()
            self.curve[it] = self.gbest_f