
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .opt_function import levy
from .optimizer_base import Optimizer

"""
Cuckoo Quantum Particle Swarm Optimization

Reference:
~~~~~~~~~~
Xiongfa Mai, Han-Bin Liu, Li-Bin Liu.
A New Hybrid Cuckoo Quantum-Behavior Particle Swarm Optimization Algorithm and its Application in Muskingum Model.
Neural Processing Letters, 2023, 55: 8309-8337.
"""
class CuckooQuantumParticleSwarmOpt(Optimizer):
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

        x = bm.random.rand(N, dim) * (ub - lb) + lb 
        fit_x = self.fun(x)[:, None]
        z = bm.random.rand(N, dim) * (ub - lb) + lb 
        fit_z = self.fun(z)[:, None]
        NS = 0
        best = bm.zeros((1, MaxIT))
        NN = int(N/2)
        for it in range(0, MaxIT):
            alpha = 1 - 0.5 * bm.log(bm.array(it + 1)) / bm.log(bm.array(MaxIT)) # Rapid Decreasing Contraction-Expansion Coefﬁcient
            delta = bm.abs((fit - bm.max(fit)) / (bm.min(fit) - bm.max(fit) + 1e-10))
            mbest = bm.sum(delta * pbest, axis=0) / N # The Weighted Mean Best Position
            phi = bm.random.rand(N, 1)
            p = phi * pbest + (1 - phi) * gbest # local attractor
            rand = bm.random.rand(N, 1)
            a = p + alpha * bm.abs(mbest - a) * bm.log(1 / bm.random.rand(N, 1)) * (1 - 2 * (rand >= 0.5)) # update
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)
            fit = self.fun(a)[:, None]

            Pa = 0.85 + 1.3 * ((it + 1) / MaxIT) ** 3 - 0.5 * (2 * (it + 1) / MaxIT) ** 2
            alpha_x = 1e-4 * MaxIT * bm.exp(bm.array((-4 * (it + 1)) / MaxIT))
            omega = bm.cos(bm.array(bm.pi * it / MaxIT) + 0.5 * bm.pi) + 1
            x_new = x + alpha_x * levy(N, dim, 1.5) * (x - gbest)
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit_x
            x, fit_x = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit_x)
            x_new = omega * x + bm.random.rand(N, 1) * bm.where((bm.random.rand(N, dim) - Pa) < 0, 0, 1) * (x[bm.random.randint(0, N, (N,))] - x[bm.random.randint(0, N, (N,))])
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit_x
            x, fit_x = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit_x)
            
            d_rand = bm.random.rand(N, 1)
            z_new = d_rand * x + (1 - d_rand) * a
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
            
            best[0, it] = gbest_f[0]
            if it > 0:
                if best[0, it] < best[0, it - 1]:
                    NS = NS + 1
                    if NS == 5:
                        index_a = bm.argsort(fit, axis=0)
                        index_x = bm.argsort(fit_x)
                        a[index_a[NN:N][:, 0]] = bm.random.rand(NN, dim) * (ub - lb) + lb 
                        x[index_x[NN:N][:, 0]] = bm.random.rand(NN, dim) * (ub - lb) + lb 
                        fit = self.fun(a)[:, None]
                        fit_x = self.fun(x)[:, None]
                        NS = 0

        return gbest, gbest_f