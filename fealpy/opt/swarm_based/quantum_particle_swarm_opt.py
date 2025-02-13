from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer

"""
Quantum Particle Swarm Optimization

"""
class QuantumParticleSwarmOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, alpha_max=0.9, alpha_min=0.4):
        fit = self.fun(self.x)
        pbest = bm.copy(self.x)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        for it in range(0, self.MaxIT):
            # exploration percentage and exploitation percentage
            self.D_pl_pt(it)

            alpha = bm.array(alpha_max - (alpha_max - alpha_min) * (it + 1) / self.MaxIT) # contraction-expansion coefficient
            mbest = bm.sum(pbest, axis=0) / self.N # average of all particle optimal position
            phi = bm.random.rand(self.N, self.dim)
            p = phi * pbest + (1 - phi) * self.gbest # local attractor
            u = bm.random.rand(self.N, self.dim)
            rand = bm.random.rand(self.N, 1)
            # update
            self.x = p + alpha * bm.abs(mbest - self.x) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], self.x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            self.update_gbest(pbest, pbest_f)
            self.curve[it] = self.gbest_f