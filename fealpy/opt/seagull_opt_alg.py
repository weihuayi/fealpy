
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Seagull Optimization Algorithm
~~~~~~~~~~
Reference:
Gaurav Dhiman, Vijay Kumar.
Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems.
Knowledge-Based Systems, 2019, 165: 169-196.
"""
class SeagullOptAlg(Optimizer):
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

        D_pl = bm.zeros((1, MaxIT))
        D_pt = bm.zeros((1, MaxIT))
        Div = bm.zeros((1, MaxIT))
        curve = bm.zeros((1, MaxIT))
        # Parameters
        Fc = 2
        u = 1
        v = 1
        for it in range(0, MaxIT):
            Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            D_pl[0, it] = 100 * Div[0, it] / bm.max(Div)
            D_pt[0, it] = 100 * bm.abs(Div[0, it] - bm.max(Div)) / bm.max(Div)

            A = Fc - (it * Fc / MaxIT) # Eq.(6)
            C = A * x # Eq.(5)
            B = 2 * A ** 2 * bm.random.rand(N, 1) # Eq.(8)
            M = B * (gbest - x) # Eq.(7)
            Ds = bm.abs(C + M) # Eq.(9)
            k = bm.random.rand(N, 1) * 2 * bm.pi
            r = u * bm.exp(v * k) # Eq.(13)
            x = r * bm.cos(k) # Eq.(10)
            y = r * bm.sin(k) # Eq.(11)
            z = r * k # Eq.(12)
            x = Ds * x * y * z + gbest # Eq.(14)
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)[:, None]
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            curve[0, it] = gbest_f[0]

        self.gbest = gbest
        self.gbest_f = gbest_f
        self.curve = curve[0]
        self.D_pl = D_pl[0]
        self.D_pt = D_pt[0]