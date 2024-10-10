
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
import random
from .optimizer_base import Optimizer
from .Levy import levy


class HippopotamusOptAlg(Optimizer):
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
        gbest_idx = bm.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = x[gbest_idx].reshape(1, -1)
        Convergence_curve = []
        for it in range(0, MaxIT):
            T = bm.exp(bm.array(it / MaxIT)) # Eq.(5)
            i1 = int(N / 2)
            I1 = bm.random.randint(1, 3, (i1, 1)) 
            I2 = bm.random.randint(1, 3, (i1, 1)) 
            Ip1 = bm.random.randint(0, 2, (i1, 1)) 
            Ip2 = bm.random.randint(0, 2, (i1, 1)) 

            # Eq.(4)
            Alfa0 = I2 * bm.random.rand(i1, dim) + Ip1
            Alfa1= 2 * bm.random.rand(i1, dim) - 1
            Alfa2 = bm.random.rand(i1, dim) 
            Alfa3 = I1 * bm.random.rand(i1, dim) + Ip2
            Alfa4 = bm.random.rand(i1, 1) * bm.ones((i1, dim))
            AA = bm.array([random.randint(0, 5) for _ in range(i1)])[:, None] 
            BB = bm.array([random.randint(0, 5) for _ in range(i1)])[:, None]
            A = (AA == 0) * Alfa0 + (AA == 1) * Alfa1 + (AA == 2) * Alfa2 + (AA == 3) * Alfa3 + (AA == 4) * Alfa4
            B = (BB == 0) * Alfa0 + (BB == 1) * Alfa1 + (BB == 2) * Alfa2 + (BB == 3) * Alfa3 + (BB == 4) * Alfa4
            RandGroupNumber = [random.randint(1, N + 1) for _ in range(i1)]
            MeanGroup = bm.zeros((i1, dim))
            for i in range(i1):
                RandGroup = bm.random.permutation(N)[: RandGroupNumber[i]]
                MeanGroup[i] = x[RandGroup].mean(axis=0)

            r1 = bm.random.rand(i1 ,1)
            X_P1 = x[: i1] + r1 * (gbest - I1 * x[: i1]) # Eq.(3)
            X_P1 = X_P1 + (lb - X_P1) * (X_P1 < lb) + (ub - X_P1) * (X_P1 > ub)
            F_P1 = self.fun(X_P1)[:, None]

            # Eq.(8)
            mask = F_P1 < fit[: i1]
            x[: i1], fit[: i1] = bm.where(mask, X_P1, x[: i1]), bm.where(mask, F_P1, fit[: i1])

            if T > 0.6:
                X_P2 = x[: i1] + A * (gbest - I2 * MeanGroup) # Eq.(6)
            else:
                r2 = bm.random.rand(i1, 1)
                # Eq.(7)
                X_P2 = ((r2 > 0.5) * 
                        (x[: i1] + B * (MeanGroup - gbest)) + 
                        (r2 <= 0.5) * 
                        (lb + bm.random.rand(i1, 1) * (ub - lb)))
            X_P2 = X_P2 + (lb - X_P2) * (X_P2 < lb) + (ub - X_P2) * (X_P2 > ub)
            F_P2 = self.fun(X_P2)[:, None]   
            
            # Eq.(9)
            mask = F_P2 < fit[: i1]
            x[: i1], fit[: i1] = bm.where(mask, X_P2, x[: i1]), bm.where(mask, F_P2, fit[: i1])

            predator = lb + bm.random.rand(i1, dim) * (ub - lb) # Eq.(10)
            F_HL = self.fun(predator)[:, None]
            distance2Leader = abs(predator - x[i1:]) # Eq.(11)
            RL = 0.05 * levy(i1, dim, 1.5) # Eq.(13)

            b = bm.random.uniform(2, 4, (i1, 1))
            c = bm.random.uniform(1, 1.5, (i1, 1))
            d = bm.random.uniform(2, 3, (i1, 1))
            g = bm.random.uniform(-1, 1, (i1, 1))

            # Eq.(12)
            X_P3 = ((fit[i1:] > F_HL) * 
                    (RL * predator + b / (c - d * bm.cos(2 * bm.pi * g)) / distance2Leader) + 
                    (fit[i1:] <= F_HL) * 
                    (RL * predator + b / (c - d * bm.cos(2 * bm.pi * g)) / (bm.random.rand(i1, dim) + 2 * distance2Leader)))
            X_P3 = X_P3 + (lb - X_P3) * (X_P3 < lb) + (ub - X_P3) * (X_P3 > ub)
            F_P3 = self.fun(X_P3)[:, None]

            # Eq.(15)
            mask = F_P3 < fit[: i1]
            x[: i1] , fit[: i1] = bm.where(mask, X_P3, x[: i1]), bm.where(mask, F_P3, fit[: i1])

            # Eq.(16)
            l_local = lb / (it + 1)
            h_local = ub / (it + 1)

            # Eq.(18)
            Blfa0 = 2 * bm.random.rand(N, dim) - 1
            Blfa1 = bm.random.rand(N, 1) * bm.ones((N, dim))
            Blfa2 = bm.random.randn(N, 1) * bm.ones((N, dim))

            DD = bm.random.randint(0, 3, N)[:, None] 
            D = (DD == 0) * Blfa0 + (DD == 1) * Blfa1 + (DD == 2) * Blfa2

            X_P4 = x + bm.random.rand() * (l_local + D * (h_local - l_local)) # Eq.(17)
            X_P4 = X_P4 + (lb - X_P4) * (X_P4 < lb) + (ub - X_P4) * (X_P4 > ub)
            F_P4 = self.fun(X_P4)[:, None]

            # Eq.(19)
            mask = F_P4 < fit
            x , fit = bm.where(mask, X_P4, x), bm.where(mask, F_P4, fit)

            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            Convergence_curve.append(gbest_f[0])
        return gbest, gbest_f, Convergence_curve
