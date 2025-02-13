from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer
"""
Improved Whale Optimization Algorithm  

Reference:
~~~~~~~~~~
Seyedali Mirjalili, Andrew Lewis.
A novel improved whale optimization algorithm to solve numerical optimization and real-world applications.
Artificial Intelligence Review, 2022, 55: 4605-4716.
"""
class ImprovedWhaleOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):

        fit = self.fun(self.x)
    
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            A = 2 * (2 - it * 2 / self.MaxIT) * bm.random.rand(self.N, 1) - (2 - it * 2 / self.MaxIT) # Eq.(3)  
            C = 2 * bm.random.rand(self.N, 1) # Eq.(4)
            p = bm.random.rand(self.N, 1)
            l = (-2 - it / self.MaxIT) * bm.random.rand(self.N, 1) + 1 # Eq.(9)
            if it < int(self.MaxIT / 2):
                rand1 = self.x[bm.random.randint(0, self.N, (self.N,))]
                rand2 = self.x[bm.random.randint(0, self.N, (self.N,))]
                rand_mean = (rand1 + rand2) / 2
                mask = bm.linalg.norm(self.x - rand1, axis=1)[:, None] < bm.linalg.norm(self.x - rand2, axis=1)[:, None]
                rand = bm.where(mask, rand2, rand1)
                # Eq.(2)
                x_new = ((p < 0.5) * (rand - A * bm.abs(C * rand - self.x)) + 
                         (p >= 0.5) * (rand_mean - A * bm.abs(C * rand_mean - self.x)))
            else:
                x_new = ((p < 0.5) * (self.gbest - A * bm.abs(C * self.gbest - self.x)) + # Eq.(6)
                         (p >= 0.5) * (bm.abs(self.gbest - self.x) * bm.exp(l) * bm.cos(l * 2 * bm.pi) + self.gbest)) # Eq.(8) 
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = (fit_new < fit)
            self.x = bm.where(mask[:, None], x_new, self.x)
            fit = bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f