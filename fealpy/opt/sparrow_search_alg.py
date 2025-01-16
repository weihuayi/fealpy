
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
        
        fit = self.fun(self.x)
        
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        best_f = fit[gbest_index]
        
        index = bm.argsort(fit)
        self.x = self.x[index]
        fit = fit[index]
        
        PDnumber = int(self.N * PD)
        SDnumber = int(self.N * SD) 
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            if bm.random.rand(1) < ST:
                self.x[0 : PDnumber] = self.x[0 : PDnumber] * bm.exp( -bm.arange(1, PDnumber + 1)[:, None] / (bm.random.rand(PDnumber, 1) * self.MaxIT))
            else:
                self.x[0 : PDnumber] = self.x[0 : PDnumber] + bm.random.randn(PDnumber, 1) * bm.ones((PDnumber, self.dim))

            self.x[PDnumber : self.N] = ((bm.arange(PDnumber, self.N)[:, None] > ((self.N - PDnumber) / 2 + PDnumber)) * 
                               (bm.random.randn(self.N - PDnumber, 1) / bm.exp((self.x[self.N - 1] - self.x[PDnumber : self.N]) / (bm.arange(PDnumber, self.N)[:, None] ** 2))) + 
                               (bm.arange(PDnumber, self.N)[:, None] <= ((self.N - PDnumber) / 2 + PDnumber)) * 
                               (self.x[0] + (bm.where(bm.random.rand(self.N - PDnumber, self.dim) < 0.5, -1, 1) / self.dim) * bm.abs(self.x[PDnumber : self.N] - self.x[0]))) 
            
            SDindex = bm.random.randint(0, self.N, (SDnumber,))
            self.x[SDindex] = ((fit[SDindex] > fit[0])[:, None] * 
                          (self.x[0] + bm.random.randn(SDnumber, 1) * bm.abs(self.x[SDindex] - self.x[0])) + 
                          (fit[SDindex] == fit[0])[:, None] * 
                          (self.x[SDindex] + (2 * bm.random.rand(SDnumber, 1) - 1) * (bm.abs(self.x[SDindex] - self.x[self.N - 1]) / (1e-8 + (fit[SDindex] - fit[self.N - 1])[:, None]))))
            
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)
            index = bm.argsort(fit)
            self.x = self.x[index]
            fit = fit[index]
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f