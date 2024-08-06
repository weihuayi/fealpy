
from fealpy.experimental.backend import backend_manager as bm 
# import numpy as bm
from fealpy.experimental.typing import TensorLike, Index, _S
from fealpy.experimental import logger
from fealpy.experimental.opt.optimizer_base import Optimizer


class HoneybadgerOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)

    # def __init__(self, option):
        # self.options = option

    '''
    @classmethod
    def get_options(
            cls,
            x0: TensorLike,
            objective,
            bm: int,
            MaxIters: int = 1000,
            MaxFunEvals: int = 10000,
            Print: bool = True,
            ) -> Problem:
        return Problem(
                x0,
                objective,
                bm=bm,
                MaxIters= MaxIters,
                MaxFunEvals=MaxFunEvals,
                Print=Print,
                )
    '''
    def run(self):

        option = self.options
        x = option["x0"]
        N =  option["NP"]
        T = option["MaxIters"]
        fit = self.fun(x)
        # fobj = option["objective"]
        dim = option["ndim"]
        lb, ub = option["domain"]
        #print("##################################", x.shape)
        # fit = fobj(x)
        #print("##################################",fit.shape)

        gbest_idx = bm.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = x[gbest_idx] 
        # print("##############",gbest)
        C = 2
        eps = 2.2204e-16
        beta = 6
        # fit, x, gbest, gbest_f = self.initialize()
        for t in range (0, T):
            alpha = C * bm.exp(bm.array(-t/T))
            S = bm.linalg.norm(x - bm.concatenate((x[-1:], x[:-1])) + eps, 2, axis=1)
            di = bm.linalg.norm(x - gbest + eps, axis=1)
            I = bm.random.rand(N) * S / (4 * bm.pi * di)
            F = bm.where(bm.random.rand(N,1) < 0.5, bm.array(1), bm.array(-1))
            # F = 2 * bm.random.randint(2, size = bm.array(N)) - 1
            # di = x - gbest
            r = bm.random.rand(N, 1)
            rand_r =  bm.random.rand(4, N, dim)
            x_new = (gbest + 
                     (r < 0.5) *  
                     (F * beta * I[:, None] * gbest +  F * alpha * rand_r[0] * (x - gbest) * bm.abs(bm.cos(2 * bm.pi * rand_r[1]) * bm.cos(2 * bm.pi * rand_r[2]))) +  
                     (r > 0.5) * 
                     (gbest + F * alpha * rand_r[3] * (x - gbest)))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            # print("###################",mask)
            x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            # print("HBA: The optimum at iteration", t + 1, "is", gbest_f) 
        return gbest, gbest_f
