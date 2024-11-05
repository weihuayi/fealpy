
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer


class BlackwingedKiteAlg(Optimizer):
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

        # Parameters
        p = 0.9

        for it in range(0, MaxIT):
            R = bm.random.rand(N, 1)
            n = 0.05 * bm.exp(bm.array(-2 * ((it / MaxIT) ** 2)))
            x_new = ((p < R) * (x + n * (1 + bm.sin(R)) * x) + (p >= R) * (x + n * (2 * R - 1) * x))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = (fit_new < fit[bm.arange(0, N)])
            x[bm.arange(0, N)], fit[bm.arange(0, N)] = bm.where(mask, x_new, x[bm.arange(0, N)]), bm.where(mask, fit_new, fit[bm.arange(0, N)])
            m = 2 * bm.sin(R + bm.pi / 2)
            fit_r = fit[bm.random.randint(0, N, (N,))]
            cauchy = bm.tan((bm.random.rand(N, dim) - 0.5) * bm.pi)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = (fit_new < fit[bm.arange(0, N)])
            x[bm.arange(0, N)], fit[bm.arange(0, N)] = bm.where(mask, x_new, x[bm.arange(0, N)]), bm.where(mask, fit_new, fit[bm.arange(0, N)])
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            pass
        return gbest, gbest_f[0]
