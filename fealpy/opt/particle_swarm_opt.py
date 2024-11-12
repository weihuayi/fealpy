
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Particle Swarm Optimization

"""

class ParticleSwarmOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):
        options = self.options
        a = options["x0"]
        N = options["NP"]
        fit = self.fun(a)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        pbest = bm.copy(a)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        gbest = pbest[gbest_index]
        gbest_f = pbest_f[gbest_index]
        c1 = 2
        c2 = 2
        w = 0.9
        v = bm.zeros((N, dim))
        vlb, ulb = 0.2 * lb, 0.2 * ub
        for it in range(0, MaxIT):
            w = 0.9 - 0.4 * (it / MaxIT)
            r1 = bm.random.rand(N, 1)
            r2 = bm.random.rand(N, 1)
            v = w * v + c1 * r1 * (pbest - a) + c2 * r2 * (gbest - a)
            v = v + (vlb - v) * (v < vlb) + (ulb - v) * (v > ulb)
            a = a + v
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)
            fit = self.fun(a)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], a, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)
            (gbest_f, gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < gbest_f else (gbest_f, gbest)
            # print("PSO: The optimum at iteration", it + 1, "is", gbest_f)
        return gbest, gbest_f
