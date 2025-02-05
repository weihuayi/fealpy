
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Butterfly Optimization Algorithm

"""
class InvasiveWeedOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, Nmax=100, Smin=0, Smax=5, n=3, sigma_initial=3, sigma_final=0.001):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            sigma = ((self.MaxIT - it) / (self.MaxIT)) ** n * (sigma_initial - sigma_final) + sigma_final
            
            Snum = bm.floor((Smax - Smin) * (fit - bm.min(fit)) / (bm.max(fit) - bm.min(fit)) + Smin)
            seed = bm.zeros((int(bm.sum(Snum)), self.dim))
            current_seed_index = 0
            for i in range(self.x.shape[0]):
                if Snum[i] > 0:
                    seed[current_seed_index:current_seed_index + int(Snum[i])] = self.x[i] + sigma * bm.random.randn(int(Snum[i]), self.dim)
                    current_seed_index += int(Snum[i])
            seed = seed + (self.lb - seed) * (seed < self.lb) + (self.ub - seed) * (seed > self.ub)
            fit_need = self.fun(seed)
            self.x = bm.concatenate([self.x, seed], axis=0)
            fit = bm.concatenate([fit, fit_need])

            index = bm.argsort(fit)
            fit = fit[index]
            self.x = self.x[index]

            if self.x.shape[0] > Nmax:
                fit = fit[:Nmax]
                self.x = self.x[:Nmax]
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f
