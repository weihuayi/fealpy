
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
        fit = self.fun(self.x)
        pbest = bm.copy(self.x)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        v = bm.zeros((self.N, self.dim))
        vlb, ulb = 0.2 * self.lb, 0.2 * self.ub
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            w = w_max - w_min * (it / self.MaxIT)
            v = w * v + c1 * bm.random.rand(self.N, self.dim) * (pbest - self.x) + c2 * bm.random.rand(self.N, self.dim) * (self.gbest - self.x)
            v = v + (vlb - v) * (v < vlb) + (ulb - v) * (v > ulb)
            self.x = self.x + v
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], self.x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            self.update_gbest(pbest, pbest_f)
            self.curve[it] = self.gbest_f