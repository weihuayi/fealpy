
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
import math
import random
import numpy as bm
from .optimizer_base import Optimizer, opt_alg_options


class CrayfishOptAlg(Optimizer):
#    def __init__(self) -> None:
#        super().__init__(problem)
    def __init__(self, option):
        self.options = option
    """
    @classmethod
    def get_options(
            cls,
            x0: TensorLike,
            objective,
            bm: int,
            domain,
            MaxIters: int = 1000,
            MaxFunEvals: int = 10000,
            Print: bool = True,
            ) -> Problem:
        return Problem(
                x0,
                objective,
                bm=bm,
                domain = domain,
                MaxIters= MaxIters,
                MaxFunEvals=MaxFunEvals,
                Print=Print,
                )
    """
    

    
    def run(self):

        #fit, x, gbest, gbest_f = self.initialize()

        option = self.options
        x = option["x0"]
        N =  option["NP"]
        T = option["MaxIters"]
        fobj = option["objective"]
        dim = option["ndim"]
        lb, ub = option["domain"]
        #print("##################################", x.shape)
        fit = fobj(x)
        #print("##################################",fit.shape)

        gbest_idx = bm.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = x[gbest_idx] 
        print("##############",gbest)

        global_position = gbest
        global_fitness = gbest_f
        for t in range(0, T):
            print("###############",t)
            C = 2 - (t / T)
            temp = bm.random.rand() * 15 + 20
            xf = (gbest + global_position) / 2
            p = 0.2 * ( 1 / (math.sqrt(2 * bm.pi) * 3)) * math.exp( - (temp - 25) ** 2 / (2 * 3 ** 2))
            rand = bm.random.rand(N, 1)
            rr = bm.random.rand(4, N, dim)
            z = [random.randint(0, N - 1) for _ in range(N)]

            #print("GGGGGGGGGGGGGGGG",gbest.shape)
            gbest = gbest.reshape(1, dim)
            P = 3 * bm.random.rand(N) * fit / (fobj(gbest) + 2.2204e-16)
            print("#################",P.shape)
            x_new = ((temp > 30) * ((rand < 0.5) * (x + C * rr[0] * (xf -x)) + 
                                    (rand > 0.5) * (x - x[z] + xf)) + 
                     (temp <= 30) * ((P > 2) * (x + bm.cos(2 * rr[1] * bm.pi) * gbest * p - bm.sin(2 * bm.pi * rr[2] * gbest * p)) + 
                                     (P[:, bm.newaxis] <= 2) * ((x - gbest) * p + p * rr[3] * x)))
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            print("PPPPPPPPPPPPPPPPPP")
            fit_new = fobj(x_new)


            mask = fit_new < fit
            x, fit = bm.where(mask[:, bm.newaxis], x_new, x), bm.where(mask, fit_new, fit)
            newbest_id = bm.argmin(fit_new)
            (global_position, global_fitness) = (x_new[newbest_id], fit_new[newbest_id]) if fit_new[newbest_id] < global_fitness else (global_position, global_fitness)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            if (t + 1) % 50 == 0:
                print("COA" + " iter" , t  + 1 , ":", gbest_f)
        return gbest, gbest_f
        
    def plot(self):
        plt.show()



