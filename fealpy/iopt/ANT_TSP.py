from fealpy.backend import backend_manager as bm
import random


def calD(citys):
    n = citys.shape[0]
    D = bm.zeros((n, n)) 
    diff = citys[:, None, :] - citys[None, :, :]
    D = bm.sqrt(bm.sum(diff ** 2, axis = -1))
    D[bm.arange(n), bm.arange(n)] = 1e-4
    return D

class Ant_TSP:
    def __init__(self, m, alpha, beta, rho, Q, Eta, D, iter_max):
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.Eta = Eta
        self.D = D
        self.iter_max = iter_max

    def cal(self, n, Tau, Table, Route_best, Length_best):
        for iter in range(0, self.iter_max):
            # 随机产生各个蚂蚁的起点城市
            start = [random.randint(0, n - 1) for _ in range(self.m)]
            Table[:, 0] = bm.array(start)    
            citys_index = bm.arange(n)

            P = bm.zeros((self.m, n - 1))

            # print('--------------------------------------')         
            for j in range(1, n):
                tabu = Table[:, :j] # 已访问的城市集合

                w = []
                for i in range(self.m):
                    tabu_set = set(tabu[i].tolist())
                    difference_list = list(set(citys_index.tolist()).difference(tabu_set))
                    w.append(difference_list)
                allow = bm.array(w)

                P = (Tau[tabu[:, -1].reshape(-1, 1), allow] ** self.alpha) * (self.Eta[tabu[:, -1].reshape(-1, 1), allow] ** self.beta)
                P /= P.sum(axis=1, keepdims=True)
                Pc = bm.cumsum(P, axis=1)
                rand_vals = bm.random.rand(self.m, 1)

                # target_index = bm.argmax(Pc >= rand_vals, axis=1)
                target_index = bm.array([bm.where(row >= rand_val)[0][0] if bm.any(row >= rand_val) else -1 for row, rand_val in zip(Pc, rand_vals.flatten())])
                Table[:, j] = allow[bm.arange(self.m), target_index]

            # 计算各个蚂蚁的路径距离
            route_indices = bm.arange(n - 1)
            Length = bm.zeros(self.m)
            Length += bm.sum(self.D[Table[:, route_indices], Table[:, route_indices + 1]], axis=1)
            Length += self.D[Table[:, -1], Table[:, 0]]
            # 计算最短路径距离及平均距离
            min_index = bm.argmin(Length)
            if iter == 0:
                Length_best[iter] = Length[min_index]
                Route_best[iter] = Table[min_index]
            else:
                min_index = bm.argmin(Length)
                Length_best[iter] = min(Length_best[iter - 1], Length[min_index])
                if Length_best[iter] == Length[min_index]:
                    Route_best[iter] = Table[min_index]
                else:
                    Route_best[iter] = Route_best[iter - 1]
            # 更新信息素
            Delta_Tau = bm.zeros((n, n))
            Delta_Tau[Table[:, :-1], Table[:, 1:]] += (self.Q / Length).reshape(-1, 1)
            Delta_Tau[Table[:, -1], Table[:, 0]] += self.Q / Length
            Tau = (1 - self.rho) * Tau + Delta_Tau

        return Length_best, Route_best


