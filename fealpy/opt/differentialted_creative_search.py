from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

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
        golden_ratio = 2 / (1 + bm.sqrt(bm.array(5)))
        ngS = int(bm.max(bm.array([6, bm.round(N * (golden_ratio / 3))])))
        pc = 0.5
        phi_qKR = 0.25 + 0.55 * ((0 + ((1 + bm.arange(0, N)) / N)) ** 0.5)[:, None] # Eq.(8)
        x_new = bm.zeros((N, dim))
        for it in range(MaxIT):
            bestInd = 0
            lamda_t = 0.1 + (0.518 * ((1 - (it / MaxIT) ** 0.5))) # Eq.(13)
            index = bm.argsort(fit, axis=0)
            fit = bm.sort(fit, axis=0)
            x = x[index[:, 0]]

            # Divergent thinking 
            eta_qKR = (bm.round(bm.random.rand(N, 1) * phi_qKR) + 1 * (bm.random.rand(N, 1) <= phi_qKR)) / 2 # Eq.(7)
            jrand = bm.floor(dim * bm.random.rand(N, 1)) # Eq.(9)
            r1 = bm.random.randint(0, N, (ngS,))
            aa = (bm.random.rand(ngS, dim) < eta_qKR[0 : ngS]) + (jrand[0 : ngS] == bm.arange(0, dim)) 
            R = bm.sin(0.5 * bm.pi * golden_ratio) * bm.tan(0.5 * bm.pi * (1 - golden_ratio * bm.random.rand(ngS, dim)))
            Y = 0.05 * bm.sign(bm.random.rand(ngS, dim) - 0.5) * bm.log(bm.random.rand(ngS, dim) / bm.random.rand(ngS, dim)) * (R ** (1 / golden_ratio)) # Eq.(17)
            x_new[0 : ngS] = (~aa * x[0 : ngS] + 
                               aa * (x[r1[0 : ngS]] + Y)) # Eq.(19)

            # Convergent thinking
            r1 = bm.random.randint(0, N, (N - ngS - 1,))
            r2 = bm.random.randint(0, N - ngS, (N - ngS - 1,)) + ngS
            aa = (bm.random.rand(N - ngS - 1, dim) < eta_qKR[ngS : N - 1]) + (jrand[ngS : N - 1] == bm.arange(0, dim))
            x_new[ngS : N - 1] = (~aa * x[ngS : N - 1] + 
                                   aa * (x[bestInd] + ((x[r2] - x[ngS : N - 1]) * lamda_t) + ((x[r1] - x[ngS : N - 1])) * bm.random.rand(N - ngS - 1, 1))) # Eq.(15)
            
            # Team diversification
            if bm.random.rand(1, 1) < pc:
                x_new[N - 1] = lb + (ub - lb) * bm.random.rand(1, dim) # Eq.(21)
            
            # Boundary handling
            # Eq.(20)
            po = x_new < lb
            x_new[po] = (x[po] + lb) / 2
            po = x_new > ub
            x_new[po] = (x[po] + ub) / 2
            
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)

            # Retrospective assessment
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit) # Eq.(22)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f) # Eq.(23)

        return gbest, gbest_f