from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer


class CrowDrinkingWaterAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, P=0.9):
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest = x[gbest_index]
        self.gbest_f = fit[gbest_index]
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))

        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            
            r = bm.random.rand(N, 1)
            x = ((r < P) * (x + bm.random.rand(N, 1) * (ub -x) + bm.random.rand(N, 1) * lb) + 
                (r >= P) * ((2 * bm.random.rand(N, dim) - 1) * (ub - lb) + lb))
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f