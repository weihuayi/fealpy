from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer


class GeneticAlgorithm(Optimizer):
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
        self.gbest = x[gbest_index]
        self.gbest_f = fit[gbest_index]

        pc = 0.7
        pm = 0.01

        index = bm.argsort(fit)
        x = x[index]
        fit = fit[index]
        self.curve = bm.zeros((MaxIT,))
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            c_num = (bm.random.rand(N, dim) > pc)
            NF = bm.ones((N, dim)) * self.gbest
            new = c_num * NF + ~c_num * x

            mum = bm.random.rand(N, dim) < pm
            new = mum * (lb + (ub - lb) * bm.random.rand(N, dim)) + ~mum * new
            new_f = self.fun(new)
            all = bm.concatenate((x, new), axis=0)
            all_f = bm.concatenate((fit, new_f))

            index = bm.argsort(all_f, axis=0)
            fit = all_f[index[0 : N]]
            x = all[index[0 : N]]
            self.update_gbest(x, fit)
            self.curve[it] = self.gbest_f