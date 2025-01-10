from .opt_function import levy
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer

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
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest_f = fit[gbest_index]

        index = bm.argsort(fit)
        self.gbest = x[index[0]]
        FSa = x[index[1 : 4]]
        FSn = x[index[4 : N]]
        Gc = 1.9
        Pdp = 0.1
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            n2 = bm.random.randint(4, N, (1,))

            index2 = bm.unique(bm.random.randint(0, N - 4, (n2[0],)))
            index3 = bm.array(list(set(bm.arange(N - 4).tolist()).difference(index2.tolist())))

            n2 = len(index2)
            n3 = N - n2 - 4

            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(3, 1)
            r1 = bm.random.rand(3, 1)
            FSa = ((r1 > Pdp) * (FSa + dg * Gc * (self.gbest - FSa)) + 
                   (r1 <= Pdp) * (lb + (ub - lb) * bm.random.rand(3, dim)))
            FSa = FSa + (lb - FSa) * (FSa < lb) + (ub - FSa) * (FSa > ub)


            r2 = bm.random.rand(n2, 1)
            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(n2, 1)
            t = bm.random.randint(0, 3, (n2,))
            FSn[index2] = ((r2 > Pdp) * (FSn[index2] + dg * Gc * (FSa[t] - FSn[index2])) + 
                           (r2 <= Pdp) * (lb + (ub - lb) * bm.random.rand(n2, dim)))
            
            r3 = bm.random.rand(n3, 1)
            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(n3, 1)
            FSn[index3] = ((r3 > Pdp) * (FSn[index3] + dg * Gc * (self.gbest - FSn[index3])) + 
                           (r3 <= Pdp) * (lb + (ub - lb) * bm.random.rand(n3, dim)))

            FSn = FSn + (lb - FSn) * (FSn < lb) + (ub - FSn) * (FSn > ub)
            Sc = bm.sum((FSa - self.gbest) ** 2)
            Smin = 10 * bm.exp(bm.array(-6)) / 365 ** (it / (MaxIT / 2.5))
            if Sc < Smin:
                FSn = FSn + 0.01 * levy(N - 4, dim, 1.5) * (ub - lb)
                FSn = FSn + (lb - FSn) * (FSn < lb) + (ub - FSn) * (FSn > ub)

            x = bm.concatenate((self.gbest[None, :], FSa, FSn), axis=0)
            fit = self.fun(x)
            index = bm.argsort(fit)
            gbest_mew = x[index[0]]
            gbest_f_mew = fit[index[0]]
            if gbest_f_mew < self.gbest_f:
                self.gbest = gbest_mew
                self.gbest_f = gbest_f_mew
            FSa = x[index[1 : 4]]
            FSn = x[index[4 : N]]
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
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        self.gbest_f = fit[gbest_index]

        index = bm.argsort(fit)
        self.gbest = x[index[0]]
        FSa = x[index[1 : 4]]
        FSa_fit = fit[index[1 : 4]]
        FSn = x[index[4 : N]]
        FSn_fit = fit[index[4 : N]]
        Gc = 1.9
        Pdp = 0.1
        dg = 0.8
        Cr = 0.5
        self.curve = bm.zeros((MaxIT))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x))/N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            n2 = bm.random.randint(4, N, (1,))

            index2 = bm.unique(bm.random.randint(0, N - 4, (n2[0],)))
            index3 = bm.array(list(set(bm.arange(N - 4).tolist()).difference(index2.tolist())))

            n2 = len(index2)
            n3 = N - n2 - 4

            r1 = bm.random.rand(3, 1)
            FSa_t = ((r1 > Pdp) * (FSa + dg * Gc * (self.gbest - FSa - bm.mean(x, axis=0))) + 
                    (r1 <= Pdp) * (lb + (ub - lb) * bm.random.rand(3, dim)))
            FSa_t = FSa_t + (lb - FSa_t) * (FSa_t < lb) + (ub - FSa_t) * (FSa_t > ub)
            mask = (bm.random.rand(3, dim) < Cr) + (bm.round(dim * bm.random.rand(3, dim)) == bm.arange(0, dim))
            Xa_t = bm.where(mask, FSa_t, FSa)
            mask = self.fun(Xa_t) < self.fun(FSa_t)
            FSa_t = bm.where(mask[:, None], Xa_t, FSa)

            r2 = bm.random.rand(n2, 1)
            t = bm.random.randint(0, 3, (n2,))
            FSn_Xt = ((r2 > Pdp) * (FSn[index2] + dg * Gc * (FSa[t] - FSn[index2])) + 
                    (r2 <= Pdp) * (lb + (ub - lb) * bm.random.rand(n2, dim)))
            FSn_Xt = FSn_Xt + (lb - FSn_Xt) * (FSn_Xt < lb) + (ub - FSn_Xt) * (FSn_Xt > ub)
            mask = (bm.random.rand(n2, dim) < Cr) + (bm.round(dim * bm.random.rand(n2, dim)) == bm.arange(0, dim))
            Xn_t = bm.where(mask, FSn_Xt, FSn[index2])
            mask = self.fun(Xn_t) < self.fun(FSn_Xt)
            FSn_Xt = bm.where(mask[:, None], Xn_t, FSn_Xt)

            r3 = bm.random.rand(n3, 1)
            FSn_Yt = ((r3 > Pdp) * (FSn[index3] + dg * Gc * (self.gbest - FSn[index3])) + 
                     (r3 <= Pdp) * (lb + (ub - lb) * bm.random.rand(n3, dim)))
            FSn_Yt = FSn_Yt + (lb - FSn_Yt) * (FSn_Yt < lb) + (ub - FSn_Yt) * (FSn_Yt > ub)
            mask = (bm.random.rand(n3, dim) < Cr) + (bm.round(dim * bm.random.rand(n3, dim)) == bm.arange(0, dim))
            Yn_t = bm.where(mask, FSn_Yt, FSn[index3])
            mask = self.fun(Yn_t) < self.fun(FSn_Yt)
            FSn_Yt = bm.where(mask[:, None], Yn_t, FSn_Yt)

            FSn_t = bm.concatenate((FSn_Xt, FSn_Yt), axis=0)
            
            gbest_t = self.gbest + dg * Gc * (self.gbest - bm.mean(FSa, axis=0))
            gbest_t = gbest_t + (lb - gbest_t) * (gbest_t < lb) + (ub - gbest_t) * (gbest_t > ub)
            mask = self.fun(gbest_t[None, :]) < self.gbest_f
            self.gbest, self.gbest_f = bm.where(mask, gbest_t, self.gbest), bm.where(mask, self.fun(gbest_t[None, :]), self.gbest_f)
            
            FS_t = bm.concatenate((self.gbest[None, :], FSa_t, FSn_t), axis=0)
            Z_fit = self.fun(FS_t)
            mask = Z_fit < fit
            x = bm.where(mask[:, None], FS_t, x)
            fit = bm.where(mask, Z_fit, fit)
            
            index = bm.argsort(fit)
            gbest_mew = x[index[0]]
            gbest_f_mew = fit[index[0]]
            if gbest_f_mew < self.gbest_f:
                self.gbest = gbest_mew
                self.gbest_f = gbest_f_mew
            FSa = x[index[1 : 4]]
            FSn = x[index[4 : N]]
            self.curve[it] = self.gbest_f