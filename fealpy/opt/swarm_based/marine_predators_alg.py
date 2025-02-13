from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer
from ..opt_function import levy
"""
Marine Predators Algorithm

Reference:
~~~~~~~~~~
Faramarzi A, Heidarinejad M, Mirjalili S, et al. 
Marine Predators Algorithm: A nature-inspired metaheuristic. 
Expert systems with applications, 2020, 152: 113377.

"""
class MarinePredatorsAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, P=0.5, FADs=0.2):
        
        fit = self.fun(self.x)
        
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        NN = int(self.N / 2)
        
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            
            CF = (1 - it / self.MaxIT) ** (2 * it / self.MaxIT)
            if it <= self.MaxIT / 3:
                RB = bm.random.randn(self.N, self.dim)
                x_new = self.x + P * bm.random.rand(self.N, self.dim) * RB * (self.gbest - RB * self.x)
            elif it > self.MaxIT / 3 and it <= 2 * self.MaxIT / 3:
                RB = bm.random.randn(NN, self.dim)
                x_new[0 : NN] = self.gbest + P * CF * RB * (RB * self.gbest - self.x[0 : NN])
                RL = 0.05 * levy(NN, self.dim, 1.5)
                x_new[NN : self.N] = self.x[NN : self.N] + P * bm.random.rand(NN, self.dim) * RL * (self.gbest - RL * self.x[NN : self.N])
            else:
                RL = 0.05 * levy(self.N, self.dim, 1.5)
                x_new = self.gbest + P * CF * RL * (RL * self.gbest - self.x)

            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            if bm.random.rand(1) < FADs:
                self.x = self.x + CF * ((self.lb + bm.random.rand(self.N, self.dim) * (self.ub - self.lb)) * (bm.random.rand(self.N, self.dim) < FADs))
                self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            else:
                self.x = self.x + (FADs * (1 - bm.random.rand(1)) + bm.random.rand(1)) * (self.x[bm.random.randint(0, self.N, (self.N,))] - self.x[bm.random.randint(0, self.N, (self.N,))])
                self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f