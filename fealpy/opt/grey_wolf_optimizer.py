from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer
"""
Grey Wolf Optimizer

"""
class GreyWolfOpt(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):
        # options = self.options
        # X = options["x0"]
        # N = options["NP"]
        fit = self.fun(self.x)
        # MaxIT = options["MaxIters"]
        # dim = options["ndim"]
        # lb, ub = options["domain"]

        X_fit_sort = bm.argsort(fit, axis=0)

        #一阶狼
        X_alpha_fit = fit[X_fit_sort[0]]
        X_alpha = self.x[X_fit_sort[0]]

        #二阶狼
        X_beta_fit = fit[X_fit_sort[1]]
        X_beta = self.x[X_fit_sort[1]]

        #三阶狼
        X_delta_fit = fit[X_fit_sort[2]]
        X_delta = self.x[X_fit_sort[2]]

        #空列表
        self.gbest_f = X_alpha_fit
        self.gbest = X_alpha
        # self.curve = bm.zeros((MaxIT,))
        # self.D_pl = bm.zeros((MaxIT,))
        # self.D_pt = bm.zeros((MaxIT,))
        # self.Div = bm.zeros((1, MaxIT))

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            # self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(X, axis=0) - X)) / N)
            # # exploration percentage and exploitation percentage
            # self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])
            
            a = 2 - 2 * it / self.MaxIT

            X1 = X_alpha - (2 * a * bm.random.rand(self.N, self.dim) - a) * bm.abs(2 * bm.random.rand(self.N, self.dim) * X_alpha - self.x)
            X2 = X_beta - (2 * a * bm.random.rand(self.N, self.dim) - a) * bm.abs(2 * bm.random.rand(self.N, self.dim) * X_beta - self.x)
            X3 = X_delta - (2 * a * bm.random.rand(self.N, self.dim) - a) * bm.abs(2 * bm.random.rand(self.N, self.dim) * X_delta - self.x)

            self.x = (X1 + X2 + X3) / 3
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            fit = self.fun(self.x)

            sort_index = bm.argsort(fit)
            X_sort = self.x[sort_index[:3]]
            fit_sort = fit[sort_index[:3]]
            
            if fit_sort[0] < X_alpha_fit:
                X_alpha, X_alpha_fit = X_sort[0], fit_sort[0]

            if X_alpha_fit < fit_sort[1] < X_beta_fit:
                X_beta, X_beta_fit = X_sort[1], fit_sort[1]
                
            if X_beta_fit < fit_sort[2] < X_delta_fit:
                X_delta, X_delta_fit = X_sort[2], fit_sort[2]

            self.gbest = X_alpha
            self.gbest_f = X_alpha_fit
            self.curve[it] = self.gbest_f
        