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
        fit = self.fun(x)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        gbest = x[gbest_index]
        gbest_f = fit[gbest_index]

        pc = 0.7
        pm = 0.01
        NN = int(N * 0.5)
        nf = bm.zeros((N, dim))
        NF = bm.zeros((N, dim))
        index = bm.argsort(fit, axis=0)
        x = x[index[:, 0]]
        fit = fit[index[:, 0]]
        curve = bm.zeros((1, MaxIT))
        # c = bm.zeros((N, dim))
        for it in range(0, MaxIT):
            c_num = (bm.random.rand(N, dim) > pc)
            NF = bm.ones((N, dim)) * gbest
            new = c_num * NF + ~c_num * x

            mum = bm.random.rand(N, dim) < pm
            new = mum * (lb + (ub - lb) * bm.random.rand(N, dim)) + ~mum * new
            new_f = self.fun(new)[:, None]
            all = bm.concatenate((x, new), axis=0)
            all_f = bm.concatenate((fit, new_f), axis=0)

            index = bm.argsort(all_f, axis=0)
            fit = all_f[index[0 : N, 0]]
            x = all[index[0 : N, 0]]
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            # curve[0, it] = gbest_f
        return gbest, gbest_f