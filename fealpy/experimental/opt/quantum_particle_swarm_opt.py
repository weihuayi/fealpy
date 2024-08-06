# import numpy as np
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer


class QuantumParticleSwarmOpt(Optimizer):
    def __init__(self, options):
        self.options = options
    

    def run(self):
        
        # parameters
        options = self.options
        a = options["x0"]
        N = options["NP"]
        fobj = options["objective"]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]

        # initialize
        # fit = self.fun(a)
        
        fit = fobj(a)
        
        pbest = bm.copy(a)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        gbest = pbest[gbest_index]
        gbest_f = pbest_f[gbest_index]
        for it in range(0, MaxIT):
            alpha = 1 - (it + 1) / (2 * MaxIT)
            mbest = sum(pbest) / N
            phi = bm.random.rand(N, dim)
            u = bm.random.rand(N, dim)
            rand = bm.random.rand(N)
            p = phi * pbest + (1 - phi) * gbest
            u = bm.random.rand(N, dim)
            rand = bm.random.rand(N, 1)
            a = p + alpha * bm.abs(mbest - a) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)
            fit = fobj(a)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], a, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)
            (gbest_f, gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < gbest_f else (gbest_f, gbest)
        return gbest, gbest_f
