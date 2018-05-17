import numpy as np
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from scipy.sparse import spdiags
from timeit import default_timer as timer

class AMGSolver():
    def __init__(self):
        pass

    def setup(self, A):
        self.A = A

    def solve(self, b):
        pass


    def coarsen_rs(self, theta=0.025):
        A = self.A
        N = A.shape[0]
        MAi = np.min(A, axis=0) # 最小的副对角线的值， 绝对值是是最大的
        D = spdiags(1/MAi, 0, N, N)
        Am = D@A

        # 删除弱连续关系
        i, j = Am.nonzero()
        val = Am[i, j]

        idx = (-val > theta) 
        As = csr_matrix((np.ones(len(idx)), (i[idx], j[idx])), 
                shape=(N, N), dtype=np.double) 
        Am = csr_matrix((val[idx], (i[idx], j[idx])), 
                shape=(N, N), dtype=np.double)
        Ass = (As + As.T)/2

        # 把孤立点放到 F 集合
        isF = np.zeros(N, dtype=np.bool)
        valence = A.sum(axis=1)
        isF[valence == 0] = True


        isC = np.zeros(N, dtype=np.bool)
        UDNode = np.arange(N)

