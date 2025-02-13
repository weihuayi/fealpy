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


"""
Differential Squirrel Search Algorithm  

Reference:
~~~~~~~~~~
Bibekananda Jena, Manoj Kumar Naik, Aneesh Wunnava, Rutuparna Panda.
A Differential Squirrel Search Algorithm.
In: Das, S., Mohanty, M.N. (eds) Advances in Intelligent Computing and Communication. Lecture Notes in Networks and Systems, vol 202. Springer, Singapore.
"""

class DifferentialSquirrelSearchAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self):
        fit = self.fun(self.x)
        
        gbest_index = bm.argmin(fit)
        self.gbest_f = fit[gbest_index]

        index = bm.argsort(fit)
        self.gbest = self.x[index[0]]
        FSa = self.x[index[1 : 4]]
        FSa_fit = fit[index[1 : 4]]
        FSn = self.x[index[4 : self.N]]
        FSn_fit = fit[index[4 : self.N]]
        Gc = 1.9
        Pdp = 0.1
        dg = 0.8
        Cr = 0.5
        
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            

            n2 = bm.random.randint(4, self.N, (1,))

            index2 = bm.unique(bm.random.randint(0, self.N - 4, (n2[0],)))
            index3 = bm.array(list(set(bm.arange(self.N - 4).tolist()).difference(index2.tolist())))

            n2 = len(index2)
            n3 = self.N - n2 - 4

            r1 = bm.random.rand(3, 1)
            FSa_t = ((r1 > Pdp) * (FSa + dg * Gc * (self.gbest - FSa - bm.mean(self.x, axis=0))) + 
                    (r1 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(3, self.dim)))
            FSa_t = FSa_t + (self.lb - FSa_t) * (FSa_t < self.lb) + (self.ub - FSa_t) * (FSa_t > self.ub)
            mask = (bm.random.rand(3, self.dim) < Cr) + (bm.round(self.dim * bm.random.rand(3, self.dim)) == bm.arange(0, self.dim))
            Xa_t = bm.where(mask, FSa_t, FSa)
            mask = self.fun(Xa_t) < self.fun(FSa_t)
            FSa_t = bm.where(mask[:, None], Xa_t, FSa)

            r2 = bm.random.rand(n2, 1)
            t = bm.random.randint(0, 3, (n2,))
            FSn_Xt = ((r2 > Pdp) * (FSn[index2] + dg * Gc * (FSa[t] - FSn[index2])) + 
                    (r2 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n2, self.dim)))
            FSn_Xt = FSn_Xt + (self.lb - FSn_Xt) * (FSn_Xt < self.lb) + (self.ub - FSn_Xt) * (FSn_Xt > self.ub)
            mask = (bm.random.rand(n2, self.dim) < Cr) + (bm.round(self.dim * bm.random.rand(n2, self.dim)) == bm.arange(0, self.dim))
            Xn_t = bm.where(mask, FSn_Xt, FSn[index2])
            mask = self.fun(Xn_t) < self.fun(FSn_Xt)
            FSn_Xt = bm.where(mask[:, None], Xn_t, FSn_Xt)

            r3 = bm.random.rand(n3, 1)
            FSn_Yt = ((r3 > Pdp) * (FSn[index3] + dg * Gc * (self.gbest - FSn[index3])) + 
                     (r3 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n3, self.dim)))
            FSn_Yt = FSn_Yt + (self.lb - FSn_Yt) * (FSn_Yt < self.lb) + (self.ub - FSn_Yt) * (FSn_Yt > self.ub)
            mask = (bm.random.rand(n3, self.dim) < Cr) + (bm.round(self.dim * bm.random.rand(n3, self.dim)) == bm.arange(0, self.dim))
            Yn_t = bm.where(mask, FSn_Yt, FSn[index3])
            mask = self.fun(Yn_t) < self.fun(FSn_Yt)
            FSn_Yt = bm.where(mask[:, None], Yn_t, FSn_Yt)

            FSn_t = bm.concatenate((FSn_Xt, FSn_Yt), axis=0)
            
            gbest_t = self.gbest + dg * Gc * (self.gbest - bm.mean(FSa, axis=0))
            gbest_t = gbest_t + (self.lb - gbest_t) * (gbest_t < self.lb) + (self.ub - gbest_t) * (gbest_t > self.ub)
            mask = self.fun(gbest_t[None, :]) < self.gbest_f
            self.gbest, self.gbest_f = bm.where(mask, gbest_t, self.gbest), bm.where(mask, self.fun(gbest_t[None, :]), self.gbest_f)
            FS_t = bm.concatenate((self.gbest[None, :], FSa_t, FSn_t), axis=0)
            Z_fit = self.fun(FS_t)
            mask = Z_fit < fit
            self.x = bm.where(mask[:, None], FS_t, self.x)
            fit = bm.where(mask, Z_fit, fit)
            
            index = bm.argsort(fit)
            gbest_mew = self.x[index[0]]
            gbest_f_mew = fit[index[0]]
            if gbest_f_mew <= self.gbest_f:
                self.gbest = gbest_mew
                self.gbest_f = gbest_f_mew
            FSa = self.x[index[1 : 4]]
            FSn = self.x[index[4 : self.N]]
            self.curve[it] = self.gbest_f