from ..opt_function import levy
from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer

"""
Squirrel Search Algorithm  

Reference:
~~~~~~~~~~
Mohit Jain, Vijander Singh, Asha Rani.
A novel nature-inspired algorithm for optimization: Squirrel search algorithm.
Swarm and Evolutionary Computation, 2019, 44: 148-175
"""


class SquirrelSearchAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self):
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest_f = fit[gbest_index]

        index = bm.argsort(fit)
        self.gbest = self.x[index[0]]
        FSa = self.x[index[1 : 4]]
        FSn = self.x[index[4 : self.N]]
        Gc = 1.9
        Pdp = 0.1
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            n2 = bm.random.randint(4, self.N, (1,))

            index2 = bm.unique(bm.random.randint(0, self.N - 4, (n2[0],)))
            index3 = bm.array(list(set(bm.arange(self.N - 4).tolist()).difference(index2.tolist())))

            n2 = len(index2)
            n3 = self.N - n2 - 4

            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(3, 1)
            r1 = bm.random.rand(3, 1)
            FSa = ((r1 > Pdp) * (FSa + dg * Gc * (self.gbest - FSa)) + 
                   (r1 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(3, self.dim)))
            FSa = FSa + (self.lb - FSa) * (FSa < self.lb) + (self.ub - FSa) * (FSa > self.ub)


            r2 = bm.random.rand(n2, 1)
            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(n2, 1)
            t = bm.random.randint(0, 3, (n2,))
            FSn[index2] = ((r2 > Pdp) * (FSn[index2] + dg * Gc * (FSa[t] - FSn[index2])) + 
                           (r2 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n2, self.dim)))
            
            r3 = bm.random.rand(n3, 1)
            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(n3, 1)
            FSn[index3] = ((r3 > Pdp) * (FSn[index3] + dg * Gc * (self.gbest - FSn[index3])) + 
                           (r3 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n3, self.dim)))

            FSn = FSn + (self.lb - FSn) * (FSn < self.lb) + (self.ub - FSn) * (FSn > self.ub)
            Sc = bm.sum((FSa - self.gbest) ** 2)
            Smin = 10 * bm.exp(bm.array(-6)) / 365 ** (it / (self.MaxIT / 2.5))
            if Sc < Smin:
                FSn = FSn + 0.01 * levy(self.N - 4, self.dim, 1.5) * (self.ub - self.lb)
                FSn = FSn + (self.lb - FSn) * (FSn < self.lb) + (self.ub - FSn) * (FSn > self.ub)

            self.x = bm.concatenate((self.gbest[None, :], FSa, FSn), axis=0)
            fit = self.fun(self.x)
            index = bm.argsort(fit)
            gbest_mew = self.x[index[0]]
            gbest_f_mew = fit[index[0]]
            if gbest_f_mew < self.gbest_f:
                self.gbest = gbest_mew
                self.gbest_f = gbest_f_mew
            FSa = self.x[index[1 : 4]]
            FSn = self.x[index[4 : self.N]]
            self.curve[it] = self.gbest_f