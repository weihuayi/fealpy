from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
import random
from .optimizer_base import Optimizer


class AntColonyOptAlg(Optimizer):
    def __init__(self, option, D) -> None:
        super().__init__(option)
        self.alpha = 1
        self.beta = 5
        self.rho = 0.5
        self.Q = 1
        self.D = D

    def run(self):
        option = self.options
        x = option["x0"]
        N = option["NP"]
        T = option["MaxIters"]
        dim = option["ndim"]
        lb, ub = option["domain"]

        Eta = 1 / (self.D + 1e-6)
        Table = bm.zeros((N, dim), dtype=int)
        Tau = bm.ones((dim, dim), dtype=bm.float64)
        route_id = bm.arange(dim - 1)

        gbest_f = float('inf')
        gbest = None

        for t in range(T):
            start = [random.randint(0, dim - 1) for _ in range(N)]
            
            Table[:, 0] = bm.array(start)
            citys_index = bm.arange(dim)

            P = bm.zeros((N, dim - 1))
            for j in range(1, dim):
                tabu = Table[:, :j]
                w = []
                for i in range(N):
                    tabu_set = set(tabu[i].tolist())
                    difference_list = list(set(citys_index.tolist()).difference(tabu_set))
                    w.append(difference_list)
                allow = bm.array(w)

                P = (Tau[tabu[:, -1].reshape(-1, 1), allow] ** self.alpha) * (Eta[tabu[:, -1].reshape(-1, 1), allow] ** self.beta)
                P /= P.sum(axis=1, keepdims=True)
                Pc = bm.cumsum(P, axis=1)
                rand_vals = bm.random.rand(N, 1)

                target_index = bm.array([bm.where(row >= rand_val)[0][0] if bm.any(row >= rand_val) else -1 for row, rand_val in zip(Pc, rand_vals.flatten())])
                Table[:, j] = allow[bm.arange(N), target_index]

            fit = bm.zeros(N)
            fit += bm.sum(self.D[Table[:, route_id], Table[:, route_id + 1]], axis=1)
            fit += self.D[Table[:, -1], Table[:, 0]]

            gbest_idx = bm.argmin(fit)
            if fit[gbest_idx] < gbest_f:
                gbest_f = fit[gbest_idx]
                gbest = Table[gbest_idx]

            Delta_Tau = bm.zeros((dim, dim))
            Delta_Tau[Table[:, :-1], Table[:, 1:]] += (self.Q / fit).reshape(-1, 1)
            Delta_Tau[Table[:, -1], Table[:, 0]] += self.Q / fit
            Tau = (1 - self.rho) * Tau + Delta_Tau

        return gbest, gbest_f

            



