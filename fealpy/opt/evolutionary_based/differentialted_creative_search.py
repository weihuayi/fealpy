from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger

from ..optimizer_base import Optimizer

"""
Differentialted Creative Search  

Reference:
~~~~~~~~~~
Poomin Duankhan, Khamron Sunat, Sirapat Chiewchanwattana, Patchara Nasa-ngium.
The Differentiated Creative Search (DCS): Leveraging differentiated knowledge-acquisition and creative realism to address complex optimization problems.
Expert Systems with Applications, 2024, 252: 123734
"""

class DifferentialtedCreativeSearch(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, pc=0.5):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        golden_ratio = 2 / (1 + bm.sqrt(bm.array(5)))
        ngS = int(bm.max(bm.array([6, bm.round(self.N * (golden_ratio / 3))])))

        phi_qKR = 0.25 + 0.55 * ((0 + ((1 + bm.arange(0, self.N)) / self.N)) ** 0.5)[:, None] # Eq.(8)
        x_new = bm.zeros((self.N, self.dim))
        for it in range(self.MaxIT):
            self.D_pl_pt(it)
                        
            bestInd = 0
            lamda_t = 0.1 + (0.518 * ((1 - (it / self.MaxIT) ** 0.5))) # Eq.(13)
            index = bm.argsort(fit)
            fit = bm.sort(fit)
            self.x = self.x[index]

            # Divergent thinking 
            eta_qKR = (bm.round(bm.random.rand(self.N, 1) * phi_qKR) + 1 * (bm.random.rand(self.N, 1) <= phi_qKR)) / 2 # Eq.(7)
            jrand = bm.floor(self.dim * bm.random.rand(self.N, 1)) # Eq.(9)
            r1 = bm.random.randint(0, self.N, (ngS,))
            aa = (bm.random.rand(ngS, self.dim) < eta_qKR[0 : ngS]) + (jrand[0 : ngS] == bm.arange(0, self.dim)) 
            R = bm.sin(0.5 * bm.pi * golden_ratio) * bm.tan(0.5 * bm.pi * (1 - golden_ratio * bm.random.rand(ngS, self.dim)))
            Y = 0.05 * bm.sign(bm.random.rand(ngS, self.dim) - 0.5) * bm.log(bm.random.rand(ngS, self.dim) / bm.random.rand(ngS, self.dim)) * (R ** (1 / golden_ratio)) # Eq.(17)
            x_new[0 : ngS] = (~aa * self.x[0 : ngS] + aa * (self.x[r1[0 : ngS]] + Y)) # Eq.(19)

            # Convergent thinking
            r1 = bm.random.randint(0, self.N, (self.N - ngS - 1,))
            r2 = bm.random.randint(0, self.N - ngS, (self.N - ngS - 1,)) + ngS
            aa = (bm.random.rand(self.N - ngS - 1, self.dim) < eta_qKR[ngS : self.N - 1]) + (jrand[ngS : self.N - 1] == bm.arange(0, self.dim))
            x_new[ngS : self.N - 1] = (~aa * self.x[ngS : self.N - 1] + 
                                        aa * 
                                        (self.x[bestInd] + ((self.x[r2] - self.x[ngS : self.N - 1]) * lamda_t) + ((self.x[r1] - self.x[ngS : self.N - 1])) * bm.random.rand(self.N - ngS - 1, 1))) # Eq.(15)
            
            # Team diversification
            if bm.random.rand(1, 1) < pc:
                x_new[self.N - 1] = self.lb + (self.ub - self.lb) * bm.random.rand(1, self.dim) # Eq.(21)
            
            # Boundary handling
            # Eq.(20)
            po = x_new < self.lb
            x_new[po] = (self.x[po] + self.lb) / 2
            po = x_new > self.ub
            x_new[po] = (self.x[po] + self.ub) / 2
            
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Retrospective assessment
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:,None], x_new, self.x), bm.where(mask, fit_new, fit) # Eq.(22)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f