from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer
from ..opt_function import levy

"""
Harris Hawks Optimization  

Reference:
~~~~~~~~~~
Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen.
Harris hawks optimization: Algorithm and applications.
Future Generation Computer Systems, 2019, 97: 849-872.
"""

class HarrisHawksOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        for it in range(self.MaxIT):
            self.D_pl_pt(it)

            E1 = 2 * (1 - (it / self.MaxIT))
            q = bm.random.rand(self.N, 1)
            Escaping_Energy = E1 * (2 * bm.random.rand(self.N, 1) - 1)
            x_rand = self.x[bm.random.randint(0, self.N, (self.N,))]
            
            x_new = ((bm.abs(Escaping_Energy) >= 1) * ((q < 0.5) * (x_rand - bm.random.rand(self.N, 1) * bm.abs(x_rand - 2 * bm.random.rand(self.N, 1) * self.x)) + 
                                                      (q >= 0.5) * (self.gbest - bm.mean(self.x, axis=0) - bm.random.rand(self.N, 1) * ((self.ub - self.lb) * bm.random.rand(self.N, 1) + self.lb))) + 
                      (bm.abs(Escaping_Energy) < 1) * (((q >= 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * (self.gbest - Escaping_Energy * bm.abs(self.gbest - self.x)) + 
                                                       ((q >= 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * (self.gbest - self.x - Escaping_Energy * bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - self.x)) + 
                                                       ((q < 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - self.x)) + 
                                                       ((q < 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - bm.mean(self.x, axis=0)))))
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            x_new = ((fit_new < fit)[:, None] * (x_new) + 
                     (fit_new >= fit)[:, None] * (((q < 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * 
                                         (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - self.x) + bm.random.rand(self.N, self.dim) * levy(self.N, self.dim, 1.5)) + 
                                         ((q < 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * 
                                         (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(self.N, 1) - 1) * self.gbest - bm.mean(self.x, axis=0)) + bm.random.rand(self.N, self.dim) * levy(self.N, self.dim, 1.5))))
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = (fit_new < fit)
            self.x = bm.where(mask[:, None], x_new, self.x)
            fit = bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f