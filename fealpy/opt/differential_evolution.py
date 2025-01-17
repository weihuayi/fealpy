
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Differential Evolution

"""
class DifferentialEvolution(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, F=0.2, CR=0.5):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            # Mutation
            v = self.x[bm.random.randint(0, self.N, (self.N,))] + F * (self.x[bm.random.randint(0, self.N, (self.N,))] - self.x[bm.random.randint(0, self.N, (self.N,))])
            
            # Crossover
            mask = bm.random.rand(self.N, self.dim) < CR
            x_new = bm.where(mask, v, self.x)

            # Boundary
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluation
            fit_new = self.fun(x_new)

            # Selection
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f
