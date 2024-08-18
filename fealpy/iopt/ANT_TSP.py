from fealpy.experimental.backend import backend_manager as bm



def calD(citys):
    n = citys.shape[0]
    D = bm.zeros((n, n)) 

    diff = citys[:, bm.newaxis, :] - citys[bm.newaxis, :, :]
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

    def cal(self):
        n = self.D.shape[0]
        Tau = bm.ones((n, n), dtype = bm.float64)
        Table = bm.zeros((m, n), dtype = int)  # 路径记录表，每一行代表一个蚂蚁走过的路径
        Route_best = bm.zeros((iter_max, n), dtype = int)  # 各代最佳路径
        Length_best = bm.zeros(iter_max)  # 各代最佳路径的长度
        for iter in range(self.iter_max):
            # 随机产生各个蚂蚁的起点城市
            start = list([random.randint(0, n - 1)for _ in range(m)])
            Table[:, 0] = bm.array(start)
            citys_index = bm.arange(n)