
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


    def run(self, N_min=120, T=2, alpha=0.2, Tf=0.8):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        NN = self.N
        
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            N = int(bm.floor(bm.array(N_min + (NN - N_min) * (1 - (it % self.MaxIT // T) / (self.MaxIT // T))))) # Eq.(3)
            gamma = 2 * bm.random.rand(self.N, 1) * (1 - it / self.MaxIT) ** (it / self.MaxIT) # Eq.(9)

            r = bm.random.rand(10, N, 1)

            x_r1 = self.x[bm.random.randint(0, N, (N,))]
            x_r2 = self.x[bm.random.randint(0, N, (N,))]
            x_r3 = self.x[bm.random.randint(0, N, (N,))]
            x_r4 = self.x[bm.random.randint(0, N, (N,))]

            y = (self.x[bm.arange(0, N)] - self.x[bm.random.randint(0, N, (N,))]) / 2 # Eq.(5)
            U1 = bm.random.randint(0, 2, (N, self.dim))
            gamma = 2 * bm.random.rand(N, 1) * (1 - it / self.MaxIT) ** (it / self.MaxIT) # Eq.(9)
            delta = 2 * bm.random.randint(0, 2, (N, self.dim)) - 1 # Eq.(8)
            S = bm.exp(fit[bm.arange(0, N)] / (bm.sum(fit) + 2.2204e-16)) # Eq.(10)
            F = bm.random.rand(N, 1) * S[:, None] * (self.x[bm.random.randint(0, N, (N,))] - self.x[bm.arange(0, N)]) / (it + 1) # Eq.(12)

            # Update population
            x_new = ((r[7] < r[8]) * ((r[5] < r[6]) * (self.x[bm.arange(0, N)] + r[0] * bm.abs(2 * r[1] * self.gbest - y)) + # Eq.(4)
                                      (r[5] >= r[6]) * ((1 - U1) * self.x[bm.arange(0, N)] + U1 * (y + r[2] * (x_r1 - x_r2)))) + # Eq.(6)  
                     (r[7] >= r[8]) * ((r[9] < Tf) * ((1 - U1) * self.x[bm.arange(0, N)] + U1 * (x_r2 + S[:, None] * (x_r3 - x_r4) - r[2] * delta * gamma * S[:, None])) + # Eq.(7)
                                       (r[9] >= Tf) * (self.gbest + (alpha * (1 - r[3]) + r[3]) * (delta * self.gbest - self.x[bm.arange(0, N)]) - r[4] * delta * gamma * F))) # Eq.(11)
            
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = (fit_new < fit[bm.arange(0, N)])
            self.x[bm.arange(0, N)], fit[bm.arange(0, N)] = bm.where(mask[:, None], x_new, self.x[bm.arange(0, N)]), bm.where(mask, fit_new, fit[bm.arange(0, N)])
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f
