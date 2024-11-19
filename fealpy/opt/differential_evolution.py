
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

        # Parameters
        F = 0.2
        CR = 0.5

        for it in range(0, MaxIT):

            # Mutation
            v = x[bm.random.randint(0, N, (N,))] + F * (x[bm.random.randint(0, N, (N,))] - x[bm.random.randint(0, N, (N,))])
            
            # Crossover
            mask = bm.random.rand(N, dim) < CR
            x_new = bm.where(mask, v, x)

            # Boundary
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)

            # Evaluation
            fit_new = self.fun(x_new)[:, None]

            # Selection
            mask = fit_new < fit
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_index = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_index], fit[gbest_index]) if fit[gbest_index] < gbest_f else (gbest, gbest_f)


        return gbest, gbest_f
