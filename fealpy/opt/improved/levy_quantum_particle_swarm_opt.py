from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer

"""
Levy Quantum Paricle Swarm Optimization (LQPSO)

Reference
~~~~~~~~~
Xiao-li Lu, Guang He. 
QPSO algorithm based on Lévy flight and its application in fuzzy portfolio.
Applied Soft Computing Journal, 2021, 99: 106894.
"""

from ..opt_function import levy
from ..opt_function import initialize

class LevyQuantumParticleSwarmOpt(Optimizer):
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
        pbest = bm.copy(x)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        self.curve = bm.zeros((MaxIT,))
        # parameters
        sigma = 0.001
        delta = 0.1

        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            # Nonlinear structure of contraction-expansion coefficient
            alpha = bm.array(0.5 + (1 - 0.5) * (1 - it / MaxIT) ** 2)
            mbest = bm.sum(pbest, axis=0) / N
            phi = bm.random.rand(N, dim)
            p = phi * pbest + (1 - phi) * self.gbest
            u = bm.random.rand(N, dim)
            rand = bm.random.rand(N, 1)
            s = levy(N, dim, 1.5) * 0.05
            x = ((u <= 0.5) * (p + alpha * bm.abs(mbest - x) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))) + 
                 (u > 0.5) * (s * bm.abs(x - p) + alpha * bm.abs(mbest - x) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))))
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            self.update_gbest(pbest, pbest_f)
            # Premature prevention mechanism
            Diversity = bm.sum(x - mbest) / (N * (ub - lb)) # Population diversity
            if Diversity < sigma:
                rand_individual = bm.random.randint(0, N, (int(delta * N),))
                x[rand_individual] = initialize(int(delta * N), dim, ub, lb)
                fit[rand_individual] = self.fun(x[rand_individual])
                mask = fit < pbest_f
                pbest, pbest_f = bm.where(mask[:, None], x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
                self.update_gbest(pbest, pbest_f)
            self.curve[it] = self.gbest_f