
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .opt_function import levy
from .optimizer_base import Optimizer

"""
Levy Quantum Butterfly Optimization Algorithm

Reference:
~~~~~~~~~~
Han-Bin Liu, Li-Bin Liu, Xiongfa Mai.
A New Hybrid Levy Quantum-Behavior Butterfly Optimization Algorithm and its Application in NL5 Muskingum Model.
Electronic Reasearch Archive, 2024, 32: 2380-2406.
"""
class LevyQuantumButterflyOptAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):
        options = self.options
        a = options["x0"]
        N = options["NP"]
        fit = self.fun(a)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        pbest = bm.copy(a)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        gbest = pbest[gbest_index]
        gbest_f = pbest_f[gbest_index]
        self.curve = bm.zeros((1, MaxIT))
        x = bm.random.rand(N, dim) * (ub - lb) + lb 
        fit_x = self.fun(x)[:, None]
        z = bm.random.rand(N, dim) * (ub - lb) + lb 
        fit_z = self.fun(z)[:, None]
        NS = 0
        best = bm.zeros((1, MaxIT))
        NN = int(N/2)
        c = 0.01
        for it in range(0, MaxIT):
            alpha = 1 - 0.5 * bm.log(bm.array(it + 1)) / bm.log(bm.array(MaxIT)) # Rapid Decreasing Contraction-Expansion Coefﬁcient
            p = 0.1 + 0.8 * it / MaxIT
            delta = bm.abs((fit - bm.max(fit)) / (bm.min(fit) - bm.max(fit) + 1e-10))
            mbest = bm.sum(delta * pbest, axis=0) / N # The Weighted Mean Best Position
            a = bm.exp(bm.array(-it))
            c = c + 0.5 / (c + MaxIT)

            f = c * (fit_x ** a)
            rand = bm.random.rand(N, 1)
            x_new = ((rand < p) * (x + ((bm.random.rand(N, 1) ** 2) * gbest - x) * f) + # Global search
                     (rand >= p) * (x + ((bm.random.rand(N, 1) ** 2) * x[bm.random.randint(0, N, (N,))] - x[bm.random.randint(0, N, (N,))]) * f)) # Local search
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit 
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)

            phi = bm.random.rand(N, 1)
            p = phi * pbest + (1 - phi) * gbest # local attractor
            u = bm.random.rand(N, 1)
            a = ((u > 0.5) * (levy(N, dim, 1.5) * bm.abs(a - p) + alpha * (mbest - a) * (1 - 2 * (bm.random.rand(N, 1) >= 0.5))) + 
                 (u <= 0.5) * (p + alpha * bm.abs(mbest - a) * bm.log(1 / bm.random.rand(N, 1)) * (1 - 2 * (bm.random.rand(N, 1) >= 0.5))))
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)
            fit = self.fun(a)[:, None]
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask, a, pbest), bm.where(mask, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)    
            (gbest_f, gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < gbest_f else (gbest_f, gbest)
            
            d_rand = bm.random.rand(N, 1)
            z_new = d_rand * a + (1 - d_rand) * x
            z_new = z_new + (lb - z_new) * (z_new < lb) + (ub - z_new) * (z_new > ub)
            fit_new = self.fun(z_new)[:, None]
            mask = fit_new < fit_z
            z, fit_z = bm.where(mask, z_new, z), bm.where(mask, fit_new, fit_z)

            mask = fit_x < fit_z
            x, fit_x = bm.where(mask, z, x), bm.where(mask, fit_z, fit_x)
            index = bm.argmin(fit_x)
            (gbest_f, gbest) = (fit_x[index], x[index]) if fit_x[index] < gbest_f else (gbest_f, gbest)

            maks = fit < fit_z
            a, fit = bm.where(maks, z, a), bm.where(maks, fit_z, fit)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask, a, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)    
            (gbest_f, gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < gbest_f else (gbest_f, gbest)
            
            gbest_new = gbest + (ub - lb) * bm.random.randn(1, dim)
            gbest_new = gbest_new + (lb - gbest_new) * (gbest_new < lb) + (ub - gbest_new) * (gbest_new > ub)
            gbest_f_new = self.fun(gbest_new)
            if gbest_f_new < gbest_f:
                gbest, gbest_f = gbest_new, gbest_f_new
            best[0, it] = gbest_f[0]
            # if it > MaxIT/2:
            #     if best[0, it-1] - best[0, it] < 1e-6:
            #         NS += 1
            #         if NS == 10:
            #             index_a = bm.argsort(fit, axis=0)
            #             index_x = bm.argsort(fit_x)
            #             a[index_a[NN:N][:, 0]] = bm.random.rand(NN, dim) * (ub - lb) + lb 
            #             x[index_x[NN:N][:, 0]] = bm.random.rand(NN, dim) * (ub - lb) + lb 
            #             fit = self.fun(a)[:, None]
            #             fit_x = self.fun(x)[:, None]
            #             NS = 0 
            self.curve[0, it] = gbest_f[0]
        self.gbest = gbest
        self.gbest_f = gbest_f
        self.curve = self.curve.flatten()