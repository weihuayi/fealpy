from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

class SineCosineAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self, a=2):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            r1 = a - it * a / self.MaxIT
            r4 = bm.random.rand(self.N, self.dim)
            
            self.x = ((r4 < 0.5) * 
                      (self.x + (r1 * bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim)) * bm.abs(2 * bm.random.rand(self.N, self.dim) * self.gbest - self.x))) + 
                      (r4 >= 0.5) * 
                      (self.x + (r1 * bm.cos(2 * bm.pi * bm.random.rand(self.N, self.dim)) * bm.abs(2 * bm.random.rand(self.N, self.dim) * self.gbest - self.x))))
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)

            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f