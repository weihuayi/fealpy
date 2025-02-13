from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer


"""
Whale Optimization Algorithm  

Reference:
~~~~~~~~~~
Seyedali Mirjalili, Andrew Lewis.
The Whale Optimization Algorithm.
Advances in Engineering Software, 2016, 95: 51-67
"""
class WhaleOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):
        
        fit = self.fun(self.x)
        
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            a = 2 - it * 2 / self.MaxIT
            a2 = -1 - it / self.MaxIT
            A = 2 * a * bm.random.rand(self.N, 1) - a
            C = 2 * bm.random.rand(self.N, 1)
            p = bm.random.rand(self.N, 1)
            l = (a2 - 1) * bm.random.rand(self.N, 1) + 1
            rand_leader_index = bm.random.randint(0, self.N, (self.N,))
            x_rand = self.x[rand_leader_index]
            self.x = ((p < 0.5) * ((bm.abs(A) >= 1) * (x_rand - A * bm.abs(C * x_rand - self.x)) + # exploration phase
                              (bm.abs(A) < 1) * (self.gbest - A * bm.abs(C * self.gbest - self.x))) + # Shrinking encircling mechanism
                (p >= 0.5) * (bm.abs(self.gbest - self.x) * bm.exp(l) * bm.cos(l * 2 * bm.pi) + self.gbest)) # Spiral updating position
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f