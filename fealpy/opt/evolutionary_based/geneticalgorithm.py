from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer


class GeneticAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, pc=0.7, pm=0.01):
        
        fit = self.fun(self.x)
        
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        index = bm.argsort(fit)
        self.x = self.x[index]
        fit = fit[index]
        
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            c_num = (bm.random.rand(self.N, self.dim) > pc)
            NF = bm.ones((self.N, self.dim)) * self.gbest
            new = c_num * NF + ~c_num * self.x

            mum = bm.random.rand(self.N, self.dim) < pm
            new = mum * (self.lb + (self.ub - self.lb) * bm.random.rand(self.N, self.dim)) + ~mum * new
            new_f = self.fun(new)
            all = bm.concatenate((self.x, new), axis=0)
            all_f = bm.concatenate((fit, new_f))

            index = bm.argsort(all_f, axis=0)
            fit = all_f[index[0 : self.N]]
            self.x = all[index[0 : self.N]]
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f