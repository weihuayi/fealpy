from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer
from sklearn.cluster import KMeans
"""
K-Means Optimization Algorithm

Reference:
~~~~~~~~~~
Changting Zhong, Gang Li, Zeng Meng, Haijiang Li, Ali Riza Yildiz, Seyedali Mirjalili.
Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers.
Neural Computing and Applications, 2024.
"""

class KmeanOptAlg(Optimizer):
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
        self.gbest = x[gbest_index]
        self.gbest_f = fit[gbest_index]

        worst_index = bm.argmax(fit)
        worst = x[worst_index]
        worst_f = fit[worst_index]

        centroid_matrix = bm.random.randn(100, 30)
        kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
        kmeans.fit(centroid_matrix)
        Centroid_matrix = kmeans.cluster_centers_

        self.curve = bm.zeros((1, MaxIT))
        self.D_pl = bm.zeros((1, MaxIT))
        self.D_pt = bm.zeros((1, MaxIT))
        self.Div = bm.zeros((1, MaxIT))
        x_new = bm.zeros((N, dim))
        for it in range(MaxIT):
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)
            # exploration percentage and exploitation percentage
            self.D_pl[0, it], self.D_pt[0, it] = self.D_pl_pt(self.Div[0, it])

            f = bm.round(bm.array((N - 4) * it / (1 - MaxIT) + N - (N - 4) / (1 - MaxIT)))
            a = 1 - (1 / MaxIT) ** (it + 1)
            mean = bm.sum(Centroid_matrix, axis=0) / 3
            
            self.curve[0, it] = self.gbest_f[0]

        self.curve = self.curve.flatten()
        self.D_pl = self.D_pl.flatten()
        self.D_pt = self.D_pt.flatten()