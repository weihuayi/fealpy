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
        fit = self.fun(x)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        gbest = x[gbest_index]
        gbest_f = fit[gbest_index]


        curve = bm.zeros((1, MaxIT))
        D_pl = bm.zeros((1, MaxIT))
        D_pt = bm.zeros((1, MaxIT))
        Div = bm.zeros((1, MaxIT))
        for it in range(MaxIT):
            Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            D_pl[0, it] = 100 * Div[0, it] / bm.max(Div)
            D_pt[0, it] = 100 * bm.abs(Div[0, it] - bm.max(Div)) / bm.max(Div)

            E1 = 2 * (1 - (it / MaxIT))
            q = bm.random.rand(N, 1)
            Escaping_Energy = E1 * (2 * bm.random.rand(N, 1) - 1)
            x_rand = x[bm.random.randint(0, N, (N,))]
            
            x_new = ((bm.abs(Escaping_Energy) >= 1) * ((q < 0.5) * (x_rand - bm.random.rand(N, 1) * bm.abs(x_rand - 2 * bm.random.rand(N, 1) * x)) + 
                                                   (q >= 0.5) * (gbest - bm.mean(x, axis=0) - bm.random.rand(N, 1) * ((ub - lb) * bm.random.rand(N, 1) + lb))) + 
                 (bm.abs(Escaping_Energy) < 1) * (((q >= 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * (gbest - Escaping_Energy * bm.abs(gbest - x)) + 
                                                  ((q >= 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * (gbest - x - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * gbest - x)) + 
                                                  ((q < 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * (gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * gbest - x)) + 
                                                  ((q < 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * (gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * gbest - bm.mean(x, axis=0)))))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            x_new = ((fit_new < fit) * (x_new) + 
                     (fit_new >= fit) * (((q < 0.5)  + (bm.abs(Escaping_Energy) >= 0.5)) * (gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * gbest - x) + bm.random.rand(N, dim) * levy(N, dim, 1.5)) + 
                                         ((q < 0.5)  + (bm.abs(Escaping_Energy) < 0.5)) * (gbest - Escaping_Energy * bm.abs(2 * (bm.random.rand(N, 1) - 1) * gbest - bm.mean(x, axis=0)) + bm.random.rand(N, dim) * levy(N, dim, 1.5))))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = (fit_new < fit)
            x = bm.where(mask, x_new, x)
            fit = bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            

            curve[0, it] = gbest_f

        self.gbest = gbest.flatten()
        self.gbest_f = gbest_f.flatten()
        self.curve = curve.flatten()
        self.D_pl = D_pl[0]
        self.D_pt = D_pt[0]