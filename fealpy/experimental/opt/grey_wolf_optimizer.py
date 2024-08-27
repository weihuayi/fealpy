from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

class GreyWolfOptimizer(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):
        options = self.options
        X = options["x0"]
        N = options["NP"]
        fit = self.fun(X)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]

        X_fit_sort = bm.argsort(fit, axis=0)

        #一阶狼
        X_alpha_fit = fit[X_fit_sort[0]]
        # rr1 = bm.argwhere(fit == X_alpha_fit)
        X_alpha = X[X_fit_sort[0]]

        #二阶狼
        X_beta_fit = fit[X_fit_sort[1]]
        # rr2 = bm.argwhere(fit == X_beta_fit)
        X_beta = X[X_fit_sort[1]]

        #三阶狼
        X_delta_fit = fit[X_fit_sort[2]]
        # rr3 = bm.argwhere(fit == X_delta_fit)
        X_delta = X[X_fit_sort[2]]

        #空列表
        gbest_f = X_alpha_fit
        gbest = X_alpha
        # Convergence_curve = []
        # Convergence_curve.append(gbest_f)

        for it in range(0, MaxIT):
            a = 2 - 2 * it / MaxIT

            A = 2 * a * bm.random.rand(N, dim) - a
            C = 2 * bm.random.rand(N, dim)
            D_alpha = bm.abs(C * X_alpha - X)
            X1 = X_alpha - A * D_alpha

            A = 2 * a * bm.random.rand(N, dim) - a
            C = 2 * bm.random.rand(N, dim)
            D_beta = bm.abs(C * X_beta - X)
            X2 = X_beta - A * D_beta
            
            A = 2 * a * bm.random.rand(N, dim) - a
            C = 2 * bm.random.rand(N, dim)
            D_delta = bm.abs(C * X_delta - X)
            X3 = X_delta - A * D_delta

            X = (X1 + X2 + X3) / 3
            X = X + (lb - X) * (X < lb) + (ub - X) * (X > ub)
            fit = self.fun(X)

            sort_index = bm.argsort(fit)
            
            X_sort1 = X[sort_index[0]]
            fit_sort1 = fit[sort_index[0]]
            
            X_sort2 = X[sort_index[1]]
            fit_sort2 = fit[sort_index[1]]
            
            X_sort3 = X[sort_index[2]]
            fit_sort3 = fit[sort_index[2]]

            if fit_sort1 < X_alpha_fit:
                X_alpha_fit = fit_sort1
                X_alpha = X_sort1

            if fit_sort2 > X_alpha_fit and fit_sort2 < X_beta_fit:
                X_beta_fit = fit_sort2
                X_beta = X_sort2

            if fit_sort3 > X_beta_fit and fit_sort3 < X_delta_fit:
                X_delta_fit = fit_sort3
                X_delta = X_sort3

            gbest = X_alpha
            gbest_f = X_alpha_fit

        return gbest, gbest_f