from .opt_function import levy
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

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
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest = x[gbest_index]
        self.gbest_f = fit[gbest_index]
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        for it in range(MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            E1 = 2 * (1 - (it / MaxIT))
            q = bm.random.rand(N, 1)
            Escaping_Energy = E1 * (2 * bm.random.rand(N, 1) - 1)
            x_rand = x[bm.random.randint(0, N, (N,))]
            
            x_new = ((bm.abs(Escaping_Energy) >= 1) * ((q < 0.5) * (x_rand - bm.random.rand(N, 1) * bm.abs(x_rand - 2 * bm.random.rand(N, 1) * x)) + 
                                                      (q >= 0.5) * (self.gbest - bm.mean(x, axis=0) - bm.random.rand(N, 1) * ((ub - lb) * bm.random.rand(N, 1) + lb))) + 
                      (bm.abs(Escaping_Energy) < 1) * (((q >= 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * (self.gbest - Escaping_Energy * bm.abs(self.gbest - x)) + 
                                                       ((q >= 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * (self.gbest - x - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * self.gbest - x)) + 
                                                       ((q < 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * self.gbest - x)) + 
                                                       ((q < 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * self.gbest - bm.mean(x, axis=0)))))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)
            x_new = ((fit_new < fit)[:, None] * (x_new) + 
                     (fit_new >= fit)[:, None] * (((q < 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * 
                                         (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * self.gbest - x) + bm.random.rand(N, dim) * levy(N, dim, 1.5)) + 
                                         ((q < 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * 
                                         (self.gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * self.gbest - bm.mean(x, axis=0)) + bm.random.rand(N, dim) * levy(N, dim, 1.5))))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)
            mask = (fit_new < fit)
            x = bm.where(mask[:, None], x_new, x)
            fit = bm.where(mask, fit_new, fit)
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f