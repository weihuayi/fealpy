from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

"""
Exponential-Trigonometric Optimization Algorithm

Reference:
~~~~~~~~~~
Tran Minh Luan, Samir Khatir, Minh Thi Tran, Bernard De Baets, Thanh Cuong-Le.
Exponential-trigonometric optimization algorithm for solving complicated engineering problems.
Computer Methods in Applied Mechanics and Engineering, 2024, 432: 117411.
"""

class ExponentialTrigonometricOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self):
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit =self.fun(x)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        gbest = x[gbest_index]
        gbest_f = fit[gbest_index]

        gbest_second = bm.zeros((1, dim))
        # parameters
        CEi = 0
        CEi_temp = 0
        UB = ub
        LB = lb
        T = bm.floor(bm.array(1 + MaxIT / 1.55)) # Eq.(2)
        CE = bm.floor(bm.array(1.2 + MaxIT / 2.25)) #Eq.(7)

        for t in range(MaxIT):
            d1 = 0.1 * bm.exp(bm.array(-0.01 * t)) * bm.cos(bm.array(0.5 * MaxIT * (1 - t / MaxIT))) # Eq.(10)
            d2 = -0.1 * bm.exp(bm.array(-0.01 * t)) * bm.cos(bm.array(0.5 * MaxIT * (1 - t / MaxIT))) # Eq.(11)
            CM = (bm.sqrt(bm.array(t / MaxIT)) ** bm.tan(d1 / (d2 + 1e-8))) * bm.random.rand(N, 1) * 0.01 # Eq.(18)
            if t == CEi:
                j = bm.random.randint(0, dim, (1,))
                r1 = bm.random.rand(1, 1)
                r2 = bm.random.rand(1, 1)
                UB = gbest[j] + (1 - t / MaxIT) * bm.abs(r2 *  gbest[j] - gbest_second[0, j]) * r1 # Eq.(3)
                LB = gbest[j] - (1 - t / MaxIT) * bm.abs(r2 *  gbest[j] - gbest_second[0, j]) * r1 # Eq.(4)
                UB = UB + (ub - UB) * (UB > ub)
                LB = LB + (lb - LB) * (LB < lb)
                CEi_temp = CEi
                CEi = 0

            q1 = bm.random.rand(N, 1)

            d1 = 0.1 * bm.exp(bm.array(0.01 * t)) * bm.cos(0.5 * MaxIT * q1) # Eq.(10)
            d2 = -0.1 * bm.exp(bm.array(0.01 * t)) * bm.cos(0.5 * MaxIT * q1) # Eq.(11)

            alpha_1 = bm.random.rand(N, dim) * 3 * (t / MaxIT - 0.85) * bm.exp(bm.abs(d1 / d2) - 1) # Eq.(9)
            alpha_2 = bm.random.rand(N, dim) * bm.exp(bm.tanh(1.5 * (-t / MaxIT - 0.75) - bm.random.rand(N, dim))) # Eq.(13) 
            alpha_3 = bm.random.rand(N, dim) * 3 * (t / MaxIT - 0.85) * bm.exp(bm.abs(d1 / d2) - 1.3) # Eq.(15)

            if t < T:
                # The first phase
                x = ((CM > 1) * (gbest + bm.random.rand(N, dim) * alpha_1 * bm.abs(gbest - x) * (1 - 2 * (q1 > 0.5))) + # Eq.(8)
                     (CM <= 1) * (gbest + bm.random.rand(N, dim) * alpha_3 * bm.abs(bm.random.rand(N, dim) * gbest - x) * (1 - 2 * (q1 > 0.5)))) # Eq.(14)
            else:
                # The second phase(
                x = ((CM > 1) * (x + bm.exp(bm.tan(bm.abs(d1 / (d2 + 1e-8)) * bm.abs(bm.random.rand(N, dim) * alpha_2 * gbest - x)))) + # Eq.(16)
                     (CM <= 1) * (x + 3 * (bm.abs(bm.random.rand(N, dim) * alpha_2 * gbest - x)) * (1 - 2 * (q1 > 0.5)))) # Eq.(12)
            x = x + (LB - x) * (x < LB) + (UB - x) * (x > UB)
            fit = self.fun(x)[:, None]
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            
            if t == CE: 
                CEi = CE + 1
                CE = CE + bm.floor(2 - t * 2 / (MaxIT - CE * 4.6) / 1) # Eq.(1)
                second_index = bm.argsort(fit)[1]
                gbest_second = x[second_index]
                

        return gbest, gbest_f