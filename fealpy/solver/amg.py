import numpy as np
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from scipy.sparse import spdiags, csr_matrix
from timeit import default_timer as timer



class AMGSolver():
    """
    代数多重网格解法器类。用代数多重网格方法求解

    Ax = b

    要从 A 图结构中生成一个抽象的网格。
    """
    def __init__(self):
        pass

    def setup(self, A):
        self.A = A

    def solve(self, b):
        pass

    def coarsen_rs(self, theta=0.025):
        A = self.A
        N = A.shape[0]

        D = A.min(axis=1).toarray()# 最小的副对角线的值， 绝对值是是最大的
        D = spdiags(1/D.reshape(-1), 0, N, N)
        Am = D@A

        # 删除弱连通关系
        i, j = Am.nonzero()
        val = np.asarray(Am[i, j]).reshape(-1)

        idx = (-val > theta)
        data = np.ones(idx.sum(), dtype=np.int)
        As = csr_matrix( # 有向图的非对称矩阵
                (data, (i[idx], j[idx])), shape=(N, N), dtype=np.int)
        Am = csr_matrix(
                (val[idx], (i[idx], j[idx])), shape=(N, N), dtype=A.dtype)
        Ass = (As + As.T)/2 # 无向图的对称矩阵

        # 把孤立点放到 F 集合
        isF = np.zeros(N, dtype=np.bool_) # 细点
        dgIn = np.asarray(As.sum(axis=1)).reshape(-1) # 接入强联接顶点的数目
        isF[digIn == 0] = True # 孤立点是细点

        # 发现一个逼近的极大独立集做为粗点
        isC = np.zeros(N, dtype=np.bool_) # C: coarse node
        U = np.arange(N)                 # U: undecided node 
        degFin = np.zeros(N, dtype=np.int) # number of F nodes strong connected to

        while isC.sum() < N/2 and U.shape[0] > 20:
            # S: selected set, changing in the coarsening
            isS = np.zeros(N, dtype=np.bool_)
            degInAll = digIn + degFin
            flag = (np.randdom.rand(N) < 0.85*degInAll/degInAll.mean()) & \
                    (degInAll > 0)
            isS[flag] = True
            S, = isS.nonzero()

            

