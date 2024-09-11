
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer


class QuantumParticleSwarmOptAlg(Optimizer):
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
        for it in range(0, MaxIT):
            alpha = 1 - (it + 1) / (2 * MaxIT)
            mbest = sum(pbest) / N
            phi = bm.random.rand(N, dim)
            p = phi * pbest + (1 - phi) * gbest
            u = bm.random.rand(N, dim)
            rand = bm.random.rand(N, 1)
            a = p + alpha * bm.abs(mbest - a) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)
            fit = self.fun(a)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], a, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)
            (gbest_f, gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < gbest_f else (gbest_f, gbest)
            # print("QPSO: The optimum at iteration", it + 1, "is", gbest_f)
        return gbest, gbest_f
