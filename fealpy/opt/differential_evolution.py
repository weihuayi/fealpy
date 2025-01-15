
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

        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            # Mutation
            v = x[bm.random.randint(0, N, (N,))] + F * (x[bm.random.randint(0, N, (N,))] - x[bm.random.randint(0, N, (N,))])
            
            # Crossover
            mask = bm.random.rand(N, dim) < CR
            x_new = bm.where(mask, v, x)

            # Boundary
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)

            # Evaluation
            fit_new = self.fun(x_new)

            # Selection
            mask = fit_new < fit
            x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f
