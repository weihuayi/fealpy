from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer


class CrowDrinkingWaterAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, P=0.9):
        fit = self.fun(self.x)

        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            r = bm.random.rand(self.N, 1)
            self.x = ((r < P) * (self.x + bm.random.rand(self.N, 1) * (self.ub -self.x) + bm.random.rand(self.N, 1) * self.lb) + 
                     (r >= P) * ((2 * bm.random.rand(self.N, self.dim) - 1) * (self.ub - self.lb) + self.lb))
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f