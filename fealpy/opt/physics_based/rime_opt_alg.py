from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer

"""
Marine Predators Algorithm

Reference:
~~~~~~~~~~
Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, Huiling Chen. 
RIME: A physics-based optimization. 
Neurocomputing, 2023, 532: 183-214.

"""
class RimeOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, w=5):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            RimeFactor = (bm.random.rand(self.N, 1) - 0.5) * 2 * bm.cos(bm.array(bm.pi * it / (self.MaxIT / 10))) * (1 - bm.round(bm.array(it * w / self.MaxIT)) / w) # Parameters of Eq.(3),(4),(5)
            E = ((it + 1) /self.MaxIT) ** 0.5 # Eq.(6)
            normalized_rime_rates = fit / (bm.linalg.norm(fit) + 1e-10) # Parameters of Eq.(7) 
            r1 = bm.random.rand(self.N, 1)
            x_new = ((r1 < E) * (self.gbest + RimeFactor * ((self.ub - self.lb) * bm.random.rand(self.N, 1) + self.lb)) + # Eq.(3)
                     (r1 >= E) * self.x)
            r2 = bm.random.rand(self.N, self.dim)
            x_new = ((r2 < normalized_rime_rates[:, None]) * (self.gbest)+ 
                     (r2 >= normalized_rime_rates[:, None]) * x_new)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f