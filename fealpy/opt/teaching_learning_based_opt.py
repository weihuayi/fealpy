from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

class TeachingLearningBasedAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            TF = bm.round(1 + bm.random.rand(self.N, self.dim))
            x_new = self.x + bm.random.rand(self.N, self.dim) * (self.gbest - TF * bm.mean(self.x, axis=0))
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            k = bm.random.randint(0, self.N, (self.N,))

            x_new = ((fit < fit[k])[:, None] * (self.x + bm.random.rand(self.N, self.dim) * (self.x - self.x[k])) + 
                     (fit >= fit[k])[:, None] * (self.x + bm.random.rand(self.N, self.dim) * (self.x[k] - self.x)))
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f