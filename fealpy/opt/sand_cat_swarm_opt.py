from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

"""
Sand Cat Swarm Optimization  

Reference:
~~~~~~~~~~
Amir Seyyedabbasi, Farzad Kiani.
Sand Cat swarm optimization: a nature-inspired algorithm to solve global optimization problems.
Engineering with Computers, 2023, 39: 2627-2651.
"""

class SandCatSwarmOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)
        
    def run(self):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(self.MaxIT):
            self.D_pl_pt(it)

            rg = 2 - 2 * it / self.MaxIT # Eq.(1)
            R = 2 * rg * bm.random.rand(self.N, 1) - rg # Eq.(2)
            r = rg * bm.random.rand(self.N, 1) # Eq.(3)
            theta = 2 * bm.pi * bm.random.rand(self.N, 1)

            self.x = ((bm.abs(R) <= 1) * (self.gbest - r * bm.abs(bm.random.rand(self.N, self.dim) * self.gbest - self.x) * bm.cos(theta)) + # Eq.(5)
                      (bm.abs(R) > 1) * (r * (self.gbest - bm.random.rand(self.N, self.dim) * self.x))) # Eq.(4)
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)

            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f