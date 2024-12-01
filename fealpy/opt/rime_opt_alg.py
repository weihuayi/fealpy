from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

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


    def run(self):
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        gbest = x[gbest_index]
        gbest_f = fit[gbest_index]
        w = 5
        # curve = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            RimeFactor = (bm.random.rand(N, 1) - 0.5) * 2 * bm.cos(bm.array(bm.pi * it / (MaxIT / 10))) * (1 - bm.round(bm.array(it * w / MaxIT)) / w) # Parameters of Eq.(3),(4),(5)
            E = (it / MaxIT) ** 0.5 # Eq.(6)
            normalized_rime_rates = fit / bm.linalg.norm(fit) # Parameters of Eq.(7) 
            r1 = bm.random.rand(N, 1)
            x_new = ((r1 < E) * (gbest + RimeFactor * ((ub - lb) * bm.random.rand(N, 1) + lb)) + # Eq.(3)
                     (r1 >= E) * x)
            r2 = bm.random.rand(N, dim)
            x_new = ((r2 < normalized_rime_rates) * (gbest)+ 
                     (r2 >= normalized_rime_rates) * x_new)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            # curve[0, it] = gbest_f
        return gbest, gbest_f