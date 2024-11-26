
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer
from .opt_function import levy
from scipy.special import gamma

def levy(n, m, beta):
    num = gamma(1 + beta) * bm.sin(bm.array(bm.pi * beta / 2))
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = bm.random.randn(n, m) * sigma_u
    v = bm.random.randn(n, m)
    z = u / (bm.abs(v) ** (1 / beta))
    return z


"""
Hippopotamus Optimization Algorithm

Reference
~~~~~~~~~
Mohammad Hussein Amiri, Nastaran Mehrabi Hashjin, Mohsen Montazeri, Seyedali Mirjalili, Nima Khodadadi. 
Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm.
Scientific reports, 2024, 14: 5032.
"""

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
        # Convergence_curve = bm.zeros([1, MaxIT])
        for it in range(0, MaxIT):
            T = bm.exp(bm.array(it / MaxIT)) # Eq.(5)
            i1 = bm.array(int(N / 2))
            I1 = bm.random.randint(1, 3, (i1, 1)) 
            I2 = bm.random.randint(1, 3, (i1, 1)) 

            # Eq.(4)
            Alfa0 = I2 * bm.random.rand(i1, dim) + bm.random.randint(0, 2, (i1, 1)) 
            Alfa1= 2 * bm.random.rand(i1, dim) - 1
            Alfa2 = bm.random.rand(i1, dim) 
            Alfa3 = I1 * bm.random.rand(i1, dim) + bm.random.randint(0, 2, (i1, 1)) 
            Alfa4 = bm.random.rand(i1, 1) * bm.ones((i1, dim))
            AA = bm.random.randint(0, 5, (i1,))[:, None]
            BB = bm.random.randint(0, 5, (i1,))[:, None]
            A = (AA == 0) * Alfa0 + (AA == 1) * Alfa1 + (AA == 2) * Alfa2 + (AA == 3) * Alfa3 + (AA == 4) * Alfa4
            B = (BB == 0) * Alfa0 + (BB == 1) * Alfa1 + (BB == 2) * Alfa2 + (BB == 3) * Alfa3 + (BB == 4) * Alfa4
            
            RandGroupNumber = bm.random.randint(1, N + 1, (i1,))
            MeanGroup = bm.zeros((i1, dim))
            for i in range(i1):
                RandGroup = bm.unique(bm.random.randint(0, N - 1, (RandGroupNumber[i],)))
                MeanGroup[i] = x[RandGroup].mean(axis=0)

            X_P1 = x[: i1] + bm.random.rand(i1 ,1) * (gbest - I1 * x[: i1]) # Eq.(3)
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

            # Eq.(12)
            X_P3 = ((fit[i1:] > F_HL) * 
                    (RL * predator + (bm.random.rand(i1, 1) * 2 + 2) / ((bm.random.rand(i1, 1) * 0.5 + 1 ) - (bm.random.rand(i1, 1) + 2) * bm.cos(2 * bm.pi * (bm.random.rand(i1, 1) * 2 - 1))) / distance2Leader) + 
                    (fit[i1:] <= F_HL) * 
                    (RL * predator + (bm.random.rand(i1, 1) * 2 + 2) / ((bm.random.rand(i1, 1) * 0.5 + 1 ) - (bm.random.rand(i1, 1) + 2) * bm.cos(2 * bm.pi * (bm.random.rand(i1, 1) * 2 - 1))) / (bm.random.rand(i1, dim) + 2 * distance2Leader)))
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

            DD = bm.random.randint(0, 3, (N,))[:, None] 
            D = (DD == 0) * Blfa0 + (DD == 1) * Blfa1 + (DD == 2) * Blfa2

            X_P4 = x + bm.random.rand(N, 1) * (l_local + D * (h_local - l_local)) # Eq.(17)
            X_P4 = X_P4 + (lb - X_P4) * (X_P4 < lb) + (ub - X_P4) * (X_P4 > ub)
            F_P4 = self.fun(X_P4)[:, None]

            # Eq.(19)
            mask = F_P4 < fit
            x , fit = bm.where(mask, X_P4, x), bm.where(mask, F_P4, fit)

            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            # Convergence_curve[0, it] = bm.copy(gbest_f[0])
        return gbest[0], gbest_f
