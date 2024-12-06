from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer
from .opt_function import levy
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
        NN = int(N / 2)
        P = 0.5
        FADs = 0.2
        x_new = bm.zeros((N, dim))
        # curve = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            CF = (1 - it / MaxIT) ** (2 * it / MaxIT)
            if it <= MaxIT / 3:
                RB = bm.random.randn(N, dim)
                x_new = x + P * bm.random.rand(N, dim) * RB * (gbest - RB * x)
            elif it > MaxIT / 3 and it <= 2 * MaxIT / 3:
                RB = bm.random.randn(NN, dim)
                x_new[0 : NN] = gbest + P * CF * RB * (RB * gbest - x[0 : NN])
                RL = 0.05 * levy(NN, dim, 1.5)
                x_new[NN : N] = x[NN : N] + P * bm.random.rand(NN, dim) * RL * (gbest - RL * x[NN : N])
            else:
                RL = 0.05 * levy(N, dim, 1.5)
                x_new = gbest + P * CF * RL * (RL * gbest - x)

            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            if bm.random.rand(1) < FADs:
                x = x + CF * ((lb + bm.random.rand(N, dim) * (ub - lb)) * (bm.random.rand(N, dim) < FADs))
                x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            else:
                x = x + (FADs * (1 - bm.random.rand(1)) + bm.random.rand(1)) * (x[bm.random.randint(0, N, (N,))] - x[bm.random.randint(0, N, (N,))])
                x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)[:, None]
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            # curve[0, it] = gbest_f
        return gbest, gbest_f