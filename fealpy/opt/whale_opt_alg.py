from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer


"""
Whale Optimization Algorithm  

Reference:
~~~~~~~~~~
Seyedali Mirjalili, Andrew Lewis.
The Whale Optimization Algorithm.
Advances in Engineering Software, 2016, 95: 51-67
"""
class WhaleOptAlg(Optimizer):
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
        curve = bm.zeros((1, MaxIT))
        D_pl = bm.zeros((1, MaxIT))
        D_pt = bm.zeros((1, MaxIT))
        Div = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            D_pl[0, it] = 100 * Div[0, it] / bm.max(Div)
            D_pt[0, it] = 100 * bm.abs(Div[0, it] - bm.max(Div)) / bm.max(Div)
            a = 2 - it * 2 / MaxIT
            a2 = -1 - it / MaxIT
            A = 2 * a * bm.random.rand(N, 1) - a
            C = 2 * bm.random.rand(N, 1)
            p = bm.random.rand(N, 1)
            l = (a2 - 1) * bm.random.rand(N, 1) + 1
            rand_leader_index = bm.random.randint(0, N, (N,))
            x_rand = x[rand_leader_index]
            x = ((p < 0.5) * ((bm.abs(A) >= 1) * (x_rand - A * bm.abs(C * x_rand - x)) + # exploration phase
                              (bm.abs(A) < 1) * (gbest - A * bm.abs(C * gbest - x))) + # Shrinking encircling mechanism
                 (p >= 0.5) * (bm.abs(gbest - x) * bm.exp(l) * bm.cos(l * 2 * bm.pi) + gbest)) # Spiral updating position
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)[:, None]
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            curve[0, it] = gbest_f
        self.gbest = gbest
        self.gbest_f = gbest_f
        self.curve = curve[0]
        self.D_pl = D_pl[0]
        self.D_pt = D_pt[0]

"""
Improved Whale Optimization Algorithm  

Reference:
~~~~~~~~~~
Seyedali Mirjalili, Andrew Lewis.
A novel improved whale optimization algorithm to solve numerical optimization and real-world applications.
Artificial Intelligence Review, 2022, 55: 4605-4716.
"""
class ImprovedWhaleOptAlg(Optimizer):
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
        curve = bm.zeros((1, MaxIT))
        D_pl = bm.zeros((1, MaxIT))
        D_pt = bm.zeros((1, MaxIT))
        Div = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            D_pl[0, it] = 100 * Div[0, it] / bm.max(Div)
            D_pt[0, it] = 100 * bm.abs(Div[0, it] - bm.max(Div)) / bm.max(Div)
            A = 2 * (2 - it * 2 / MaxIT) * bm.random.rand(N, 1) - (2 - it * 2 / MaxIT) # Eq.(3)  
            C = 2 * bm.random.rand(N, 1) # Eq.(4)
            p = bm.random.rand(N, 1)
            l = (-2 - it / MaxIT) * bm.random.rand(N, 1) + 1 # Eq.(9)
            if it < int(MaxIT / 2):
                rand1 = x[bm.random.randint(0, N, (N,))]
                rand2 = x[bm.random.randint(0, N, (N,))]
                rand_mean = (rand1 + rand2) / 2
                mask = bm.linalg.norm(x - rand1, axis=1)[:, None] < bm.linalg.norm(x - rand2, axis=1)[:, None]
                rand = bm.where(mask, rand2, rand1)
                # Eq.(2)
                x_new = ((p < 0.5) * (rand - A * bm.abs(C * rand - x)) + 
                         (p >= 0.5) * (rand_mean - A * bm.abs(C * rand_mean - x)))
            else:
                x_new = ((p < 0.5) * (gbest - A * bm.abs(C * gbest - x)) + # Eq.(6)
                         (p >= 0.5) * (bm.abs(gbest - x) * bm.exp(l) * bm.cos(l * 2 * bm.pi) + gbest)) # Eq.(8) 
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = (fit_new < fit)
            x = bm.where(mask, x_new, x)
            fit = bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            curve[0, it] = gbest_f
        self.gbest = gbest
        self.gbest_f = gbest_f[0]
        self.curve = curve[0]
        self.D_pl = D_pl[0]
        self.D_pt = D_pt[0]