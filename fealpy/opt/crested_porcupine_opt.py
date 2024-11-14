
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Crested Porcupine Optimization

Reference
~~~~~~~~~
Mohamed Abdel-Basset, Reda Mohamed, Mohamed Abouhawwash.
Crested Porcupine Optimizer: A new nature-inspired metaheuristic.
Knowledge-Based Systems, 2024, 284: 111257
"""
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

        # Parameters
        N_min = 120
        T = 2
        alpha = 0.2
        Tf= 0.8
        NN = N

        for it in range(0, MaxIT):
            N = int(bm.floor(bm.array(N_min + (NN - N_min) * (1 - (it % MaxIT // T) / (MaxIT // T))))) # Eq.(3)
            gamma = 2 * bm.random.rand(N, 1) * (1 - it / MaxIT) ** (it / MaxIT) # Eq.(9)

            r = bm.random.rand(10, N, 1)

            x_r1 = x[bm.random.randint(0, N, (N,))]
            x_r2 = x[bm.random.randint(0, N, (N,))]
            x_r3 = x[bm.random.randint(0, N, (N,))]
            x_r4 = x[bm.random.randint(0, N, (N,))]

            y = (x[bm.arange(0, N)] - x[bm.random.randint(0, N, (N,))]) / 2 # Eq.(5)
            U1 = bm.random.randint(0, 2, (N, dim))
            gamma = 2 * bm.random.rand(N, 1) * (1 - it / MaxIT) ** (it / MaxIT) # Eq.(9)
            delta = 2 * bm.random.randint(0, 2, (N, dim)) - 1 # Eq.(8)
            S = bm.exp(fit[bm.arange(0, N)] / (bm.sum(fit) + 2.2204e-16)) # Eq.(10)
            F = bm.random.rand(N, 1) * S * (x[bm.random.randint(0, N, (N,))] - x[bm.arange(0, N)]) / (it + 1) # Eq.(12)

            # Update population
            x_new = ((r[7] < r[8]) * ((r[5] < r[6]) * (x[bm.arange(0, N)] + r[0] * bm.abs(2 * r[1] * gbest - y)) + # Eq.(4)
                                      (r[5] >= r[6]) * ((1 - U1) * x[bm.arange(0, N)] + U1 * (y + r[2] * (x_r1 - x_r2)))) + # Eq.(6)  
                     (r[7] >= r[8]) * ((r[9] < Tf) * ((1 - U1) * x[bm.arange(0, N)] + U1 * (x_r2 + S * (x_r3 - x_r4) - r[2] * delta * gamma * S)) + # Eq.(7)
                                       (r[9] >= Tf) * (gbest + (alpha * (1 - r[3]) + r[3]) * (delta * gbest - x[bm.arange(0, N)]) - r[4] * delta * gamma * F))) # Eq.(11)
            
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = (fit_new < fit[bm.arange(0, N)])
            x[bm.arange(0, N)], fit[bm.arange(0, N)] = bm.where(mask, x_new, x[bm.arange(0, N)]), bm.where(mask, fit_new, fit[bm.arange(0, N)])
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            
        return gbest, gbest_f[0]
