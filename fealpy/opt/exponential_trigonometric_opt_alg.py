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
        fit =self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest = x[gbest_index]
        self.gbest_f = fit[gbest_index]
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        gbest_second = bm.zeros((dim,))
        # parameters
        CEi = 0
        CEi_temp = 0
        UB = ub
        LB = lb
        T = bm.floor(bm.array(1 + MaxIT / 1.55)) # Eq.(2)
        CE = bm.floor(bm.array(1.2 + MaxIT / 2.25)) #Eq.(7)

        for it in range(MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            
            d1 = 0.1 * bm.exp(bm.array(-0.01 * (it + 1))) * bm.cos(bm.array(0.5 * MaxIT * (1 - it / MaxIT))) # Eq.(10)
            d2 = -0.1 * bm.exp(bm.array(-0.01 * (it + 1))) * bm.cos(bm.array(0.5 * MaxIT * (1 - it / MaxIT))) # Eq.(11)
            CM = (bm.sqrt(bm.array((it + 1) / MaxIT)) ** bm.tan(d1 / (d2 + 1e-8))) * bm.random.rand(N, 1) * 0.01 # Eq.(18)
            if it == CEi:
                j = bm.random.randint(0, dim, (1,))
                r1 = bm.random.rand(1, 1)
                r2 = bm.random.rand(1, 1)
                UB = self.gbest[j] + (1 - it / MaxIT) * bm.abs(r2 *  self.gbest[j] - gbest_second[j]) * r1 # Eq.(3)
                LB = self.gbest[j] - (1 - it / MaxIT) * bm.abs(r2 *  self.gbest[j] - gbest_second[j]) * r1 # Eq.(4)
                UB = UB + (ub - UB) * (UB > ub)
                LB = LB + (lb - LB) * (LB < lb)
                CEi_temp = CEi
                CEi = 0

            q1 = bm.random.rand(N, 1)

            d1 = 0.1 * bm.exp(bm.array(0.01 * it)) * bm.cos(0.5 * MaxIT * q1) # Eq.(10)
            d2 = -0.1 * bm.exp(bm.array(0.01 * it)) * bm.cos(0.5 * MaxIT * q1) # Eq.(11)

            alpha_1 = bm.random.rand(N, dim) * 3 * (it / MaxIT - 0.85) * bm.exp(bm.abs(d1 / d2) - 1) # Eq.(9)
            alpha_2 = bm.random.rand(N, dim) * bm.exp(bm.tanh(1.5 * (-it / MaxIT - 0.75) - bm.random.rand(N, dim))) # Eq.(13) 
            alpha_3 = bm.random.rand(N, dim) * 3 * (it / MaxIT - 0.85) * bm.exp(bm.abs(d1 / d2) - 1.3) # Eq.(15)

            if it < T:
                # The first phase
                x = ((CM > 1) * (self.gbest + bm.random.rand(N, dim) * alpha_1 * bm.abs(self.gbest - x) * (1 - 2 * (q1 > 0.5))) + # Eq.(8)
                     (CM <= 1) * (self.gbest + bm.random.rand(N, dim) * alpha_3 * bm.abs(bm.random.rand(N, dim) * self.gbest - x) * (1 - 2 * (q1 > 0.5)))) # Eq.(14)
            else:
                # The second phase(
                x = ((CM > 1) * (x + bm.exp(bm.tan(bm.abs(d1 / (d2 + 1e-8)) * bm.abs(bm.random.rand(N, dim) * alpha_2 * self.gbest - x)))) + # Eq.(16)
                     (CM <= 1) * (x + 3 * (bm.abs(bm.random.rand(N, dim) * alpha_2 * self.gbest - x)) * (1 - 2 * (q1 > 0.5)))) # Eq.(12)
            CEi = CEi_temp
            x = x + (LB - x) * (x < LB) + (UB - x) * (x > UB)
            fit = self.fun(x)
            self.update_gbest(x, fit)
            
            if it == CE: 
                CEi = CE + 1
                CE = CE + bm.floor(2 - it * 2 / (MaxIT - CE * 4.6) / 1) # Eq.(1)
                second_index = bm.argsort(fit)[1]
                gbest_second = x[second_index]
            self.curve[it] = self.gbest_f
