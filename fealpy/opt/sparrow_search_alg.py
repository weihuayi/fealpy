
from .opt_function import levy
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Sparrow Search Algorithm  

Reference:
~~~~~~~~~~
Mohit Jain, Vijander Singh, Asha Rani.
A novel swarm intelligence optimization approach: sparrow search algorithm.
Systems Science & Control Engineering, 2020, 8: 22-34.
"""


class SparrowSearchAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self, ST=0.6, PD=0.7, SD=0.2):
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
        best_f = fit[gbest_index]
        # x_new = bm.zeros((N, dim))
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))

        index = bm.argsort(fit)
        x = x[index]
        fit = fit[index]
        
        PDnumber = int(N * PD)
        SDnumber = int(N * SD) 
        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            if bm.random.rand(1) < ST:
                x[0 : PDnumber] = x[0 : PDnumber] * bm.exp( -bm.arange(1, PDnumber + 1)[:, None] / (bm.random.rand(PDnumber, 1) * MaxIT))
            else:
                x[0 : PDnumber] = x[0 : PDnumber] + bm.random.randn(PDnumber, 1) * bm.ones((PDnumber, dim))

            x[PDnumber : N] = ((bm.arange(PDnumber, N)[:, None] > ((N - PDnumber) / 2 + PDnumber)) * 
                               (bm.random.randn(N - PDnumber, 1) / bm.exp((x[N - 1] - x[PDnumber : N]) / (bm.arange(PDnumber, N)[:, None] ** 2))) + 
                               (bm.arange(PDnumber, N)[:, None] <= ((N - PDnumber) / 2 + PDnumber)) * 
                               (x[0] + (bm.where(bm.random.rand(N - PDnumber, dim) < 0.5, -1, 1) / dim) * bm.abs(x[PDnumber : N] - x[0]))) 
            
            SDindex = bm.random.randint(0, N, (SDnumber,))
            x[SDindex] = ((fit[SDindex] > fit[0])[:, None] * 
                          (x[0] + bm.random.randn(SDnumber, 1) * bm.abs(x[SDindex] - x[0])) + 
                          (fit[SDindex] == fit[0])[:, None] * 
                          (x[SDindex] + (2 * bm.random.rand(SDnumber, 1) - 1) * (bm.abs(x[SDindex] - x[N - 1]) / (1e-8 + (fit[SDindex] - fit[N - 1])[:, None]))))
            
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)
            index = bm.argsort(fit)
            x = x[index]
            fit = fit[index]
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f