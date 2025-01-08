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
        fit = self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest = x[gbest_index]
        self.gbest_f = fit[gbest_index]
        NN = int(N / 2)
        P = 0.5
        FADs = 0.2 
        
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            
            CF = (1 - it / MaxIT) ** (2 * it / MaxIT)
            if it <= MaxIT / 3:
                RB = bm.random.randn(N, dim)
                x_new = x + P * bm.random.rand(N, dim) * RB * (self.gbest - RB * x)
            elif it > MaxIT / 3 and it <= 2 * MaxIT / 3:
                RB = bm.random.randn(NN, dim)
                x_new[0 : NN] = self.gbest + P * CF * RB * (RB * self.gbest - x[0 : NN])
                RL = 0.05 * levy(NN, dim, 1.5)
                x_new[NN : N] = x[NN : N] + P * bm.random.rand(NN, dim) * RL * (self.gbest - RL * x[NN : N])
            else:
                RL = 0.05 * levy(N, dim, 1.5)
                x_new = self.gbest + P * CF * RL * (RL * self.gbest - x)

            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
            self.update_gbest(x, fit)
            if bm.random.rand(1) < FADs:
                x = x + CF * ((lb + bm.random.rand(N, dim) * (ub - lb)) * (bm.random.rand(N, dim) < FADs))
                x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            else:
                x = x + (FADs * (1 - bm.random.rand(1)) + bm.random.rand(1)) * (x[bm.random.randint(0, N, (N,))] - x[bm.random.randint(0, N, (N,))])
                x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f