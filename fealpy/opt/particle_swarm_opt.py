
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


    def run(self, c1=2, c2=2, w_max=0.9, w_min=0.4):
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
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        v = bm.zeros((N, dim))
        vlb, ulb = 0.2 * lb, 0.2 * ub
        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(a, axis=0) - a))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            
            w = w_max - w_min * (it / MaxIT)
            v = w * v + c1 * bm.random.rand(N, dim) * (pbest - a) + c2 * bm.random.rand(N, dim) * (self.gbest - a)
            v = v + (vlb - v) * (v < vlb) + (ulb - v) * (v > ulb)
            a = a + v
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)
            fit = self.fun(a)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], a, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            self.update_gbest(pbest, pbest_f)
            self.curve[it] = self.gbest_f