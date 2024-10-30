
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer


class CrestedPorcupineOpt(Optimizer):
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
        N_min = 120
        T = 2
        alpha = 0.2
        Tf= 0.8
        NN = N
        for it in range(0, MaxIT):
            N = bm.floor(bm.array(N_min + (NN - N_min) * (1 - (it % MaxIT // T) / (MaxIT // T))))
            r8 = bm.random.rand(int(N), 1)
            r9 = bm.random.rand(int(N), 1)
            r6 = bm.random.rand(int(N), 1)
            r7 = bm.random.rand(int(N), 1)
            r10 = bm.random.rand(int(N), 1)
            r1 = bm.random.randn(int(N), 1)
            r2 = bm.random.rand(int(N), 1)
            r3 = bm.random.rand(int(N), 1)
            r4 = bm.random.rand(int(N), 1)
            r5 = bm.random.rand(int(N), 1)
            r6 = bm.random.rand(int(N), 1)
            index = bm.random.randint(0, int(N), (int(N),))
            index1 = bm.random.randint(0, int(N), (int(N),))
            index2 = bm.random.randint(0, int(N), (int(N),))
            index3 = bm.random.randint(0, int(N), (int(N),))
            index4 = bm.random.randint(0, int(N), (int(N),))
            x_r = x[index]
            x_r1 = x[index1]
            x_r2 = x[index2]
            x_r3 = x[index3]
            x_r4 = x[index4]
            y = (x[bm.arange(0, int(N))] - x_r) / 2
            U1 = bm.random.randint(0, 2, (int(N), dim))
            gamma = 2 * bm.random.rand(int(N), 1) * (1 - it / MaxIT) ** (it / MaxIT)
            delta = bm.random.randint(0, 2, (int(N), dim))

            S = bm.exp(fit[bm.arange(0, int(N))] / (bm.sum(fit) + 2.2204e-16))
            m = bm.exp(fit[bm.arange(0, int(N))] / (bm.sum(fit) + 2.2204e-16))
            F = bm.random.rand(int(N), 1) * m * (x_r - x[bm.arange(0, int(N))]) / (it + 1)
            x_new = ((r8 < r9) * ((r6 < r7) * (x[bm.arange(0, int(N))] + r1 * bm.abs(2 * r2 * gbest - y)) + 
                                  (r6 >= r7) * ((1 - U1) * x[bm.arange(0, int(N))] + U1 * (y + r3 * (x_r1 - x_r2)))) +  
                     (r8 >= r9) * ((r10 < Tf) * ((1 - U1) * x[bm.arange(0, int(N))] + U1 * (x_r2 + S * (x_r3 - x_r4) - r3 * delta * gamma * S)) + 
                                   (r10 >= Tf) * (gbest + (alpha * (1 - r4) + r4) * (delta * gbest - x[bm.arange(0, int(N))]) - r5 * delta * gamma * F)))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = (fit_new < fit[bm.arange(0, int(N))])
            
            x[bm.arange(0, int(N))], fit[bm.arange(0, int(N))] = bm.where(mask, x_new, x[bm.arange(0, int(N))]), bm.where(mask, fit_new, fit[bm.arange(0, int(N))])
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            
        return gbest, gbest_f
