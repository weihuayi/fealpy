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
        # options = self.options
        # x = options["x0"]
        # N = options["NP"]
        fit = self.fun(self.x)
        # MaxIT = options["MaxIters"]
        # dim = options["ndim"]
        # lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]


        # self.curve = bm.zeros((MaxIT,))
        # self.D_pl = bm.zeros((MaxIT,))
        # self.D_pt = bm.zeros((MaxIT,))
        # self.Div = bm.zeros((1, MaxIT))

        for it in range(self.MaxIT):
            self.D_pl_pt(it)
            # self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

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