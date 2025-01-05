
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
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        c1 = 2
        c2 = 2
        w = 0.9
        v = bm.zeros((N, dim))
        vlb, ulb = 0.2 * lb, 0.2 * ub
        self.curve = bm.zeros((1, MaxIT))
        self.D_pl = bm.zeros((1, MaxIT))
        self.D_pt = bm.zeros((1, MaxIT))
        self.Div = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(a, axis=0) - a))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[0, it], self.D_pt[0, it] = self.D_pl_pt(self.Div[0, it])
            
            w = 0.9 - 0.4 * (it / MaxIT)
            v = w * v + c1 * bm.random.rand(N, dim) * (pbest - a) + c2 * bm.random.rand(N, dim) * (self.gbest - a)
            v = v + (vlb - v) * (v < vlb) + (ulb - v) * (v > ulb)
            a = a + v
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)
            fit = self.fun(a)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], a, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)
            (self.gbest_f, self.gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < self.gbest_f else (self.gbest_f, self.gbest)
            self.curve[0, it] = self.gbest_f
        
        self.D_pl = self.D_pl.flatten()
        self.D_pt = self.D_pt.flatten()
