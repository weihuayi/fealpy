
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
        fit = self.fun(self.x)
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx].reshape(1, -1)
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            T = bm.exp(bm.array(it / self.MaxIT)) # Eq.(5)
            i1 = bm.array(int(self.N / 2))
            I1 = bm.random.randint(1, 3, (i1, 1)) 
            I2 = bm.random.randint(1, 3, (i1, 1)) 

            # Eq.(4)
            Alfa0 = I2 * bm.random.rand(i1, self.dim) + bm.random.randint(0, 2, (i1, 1)) 
            Alfa1= 2 * bm.random.rand(i1, self.dim) - 1
            Alfa2 = bm.random.rand(i1, self.dim) 
            Alfa3 = I1 * bm.random.rand(i1, self.dim) + bm.random.randint(0, 2, (i1, 1)) 
            Alfa4 = bm.random.rand(i1, 1) * bm.ones((i1, self.dim))
            AA = bm.random.randint(0, 5, (i1,))[:, None]
            BB = bm.random.randint(0, 5, (i1,))[:, None]
            A = (AA == 0) * Alfa0 + (AA == 1) * Alfa1 + (AA == 2) * Alfa2 + (AA == 3) * Alfa3 + (AA == 4) * Alfa4
            B = (BB == 0) * Alfa0 + (BB == 1) * Alfa1 + (BB == 2) * Alfa2 + (BB == 3) * Alfa3 + (BB == 4) * Alfa4
            
            RandGroupNumber = bm.random.randint(1, self.N + 1, (i1,))
            MeanGroup = bm.zeros((i1, self.dim))
            for i in range(i1):
                RandGroup = bm.unique(bm.random.randint(0, self.N - 1, (RandGroupNumber[i],)))
                MeanGroup[i] = self.x[RandGroup].mean(axis=0)

            X_P1 = self.x[: i1] + bm.random.rand(i1 ,1) * (self.gbest - I1 * self.x[: i1]) # Eq.(3)
            X_P1 = X_P1 + (self.lb - X_P1) * (X_P1 < self.lb) + (self.ub - X_P1) * (X_P1 > self.ub)
            F_P1 = self.fun(X_P1)

            # Eq.(8)
            mask = F_P1 < fit[: i1]
            self.x[: i1], fit[: i1] = bm.where(mask[:, None], X_P1, self.x[: i1]), bm.where(mask, F_P1, fit[: i1])

            if T > 0.6:
                X_P2 = self.x[: i1] + A * (self.gbest - I2 * MeanGroup) # Eq.(6)
            else:
                r2 = bm.random.rand(i1, 1)
                # Eq.(7)
                X_P2 = ((r2 > 0.5) * 
                        (self.x[: i1] + B * (MeanGroup - self.gbest)) + 
                        (r2 <= 0.5) * 
                        (self.lb + bm.random.rand(i1, 1) * (self.ub - self.lb)))
            X_P2 = X_P2 + (self.lb - X_P2) * (X_P2 < self.lb) + (self.ub - X_P2) * (X_P2 > self.ub)
            F_P2 = self.fun(X_P2)  
            
            # Eq.(9)
            mask = F_P2 < fit[: i1]
            self.x[: i1], fit[: i1] = bm.where(mask[:, None], X_P2, self.x[: i1]), bm.where(mask, F_P2, fit[: i1])

            predator = self.lb + bm.random.rand(i1, self.dim) * (self.ub - self.lb) # Eq.(10)
            F_HL = self.fun(predator)
            distance2Leader = abs(predator - self.x[i1:]) # Eq.(11)
            RL = 0.05 * levy(i1, self.dim, 1.5) # Eq.(13)

            # Eq.(12)
            X_P3 = ((fit[i1:] > F_HL)[:, None] * 
                    (RL * predator + (bm.random.rand(i1, 1) * 2 + 2) / ((bm.random.rand(i1, 1) * 0.5 + 1 ) - 
                    (bm.random.rand(i1, 1) + 2) * bm.cos(2 * bm.pi * (bm.random.rand(i1, 1) * 2 - 1))) / distance2Leader) + 
                    (fit[i1:] <= F_HL)[:, None] * 
                    (RL * predator + (bm.random.rand(i1, 1) * 2 + 2) / ((bm.random.rand(i1, 1) * 0.5 + 1 ) - 
                    (bm.random.rand(i1, 1) + 2) * bm.cos(2 * bm.pi * (bm.random.rand(i1, 1) * 2 - 1))) / (bm.random.rand(i1, self.dim) + 2 * distance2Leader)))
            X_P3 = X_P3 + (self.lb - X_P3) * (X_P3 < self.lb) + (self.ub - X_P3) * (X_P3 > self.ub)
            F_P3 = self.fun(X_P3)

            # Eq.(15)
            mask = F_P3 < fit[: i1]
            self.x[: i1] , fit[: i1] = bm.where(mask[:, None], X_P3, self.x[: i1]), bm.where(mask, F_P3, fit[: i1])

            # Eq.(16)
            l_local = self.lb / (it + 1)
            h_local = self.ub / (it + 1)

            # Eq.(18)
            Blfa0 = 2 * bm.random.rand(self.N, self.dim) - 1
            Blfa1 = bm.random.rand(self.N, 1) * bm.ones((self.N, self.dim))
            Blfa2 = bm.random.randn(self.N, 1) * bm.ones((self.N, self.dim))

            DD = bm.random.randint(0, 3, (self.N,))[:, None] 
            D = (DD == 0) * Blfa0 + (DD == 1) * Blfa1 + (DD == 2) * Blfa2

            X_P4 = self.x + bm.random.rand(self.N, 1) * (l_local + D * (h_local - l_local)) # Eq.(17)
            X_P4 = X_P4 + (self.lb - X_P4) * (X_P4 < self.lb) + (self.ub - X_P4) * (X_P4 > self.ub)
            F_P4 = self.fun(X_P4)

            # Eq.(19)
            mask = F_P4 < fit
            self.x , fit = bm.where(mask[:, None], X_P4, self.x), bm.where(mask, F_P4, fit)

            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f