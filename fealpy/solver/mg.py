import numpy as np
from scipy.sparse import spdiags, tril, triu
from scipy.sparse.linalg import cg, dsolve, spsolve

class GaussSeidelSmoother():
    def __init__(self, A):
        """

        Notes
        -----

        对称正定矩阵的 Gauss 光滑

        """

        self.L0 = tril(A).tocsr()
        self.U0 = triu(A, k=1).tocsr()

        self.U1 = self.L0.T.tocsr()
        self.L1 = self.U0.T.tocsr()

    def smooth(self, b, x0, lower=True, maxit=3):
        if lower:
            for i in range(maxit):
                #x0[:] = spsolve_triangular(self.L0, b-self.U0@x0, lower=lower)
                x0[:] = spsolve(self.L0, b-self.U0@x0, permc_spec="NATURAL")
        else:
            for i in range(maxit):
                #x0[:] = spsolve_triangular(self.U1, b-self.L1@x0, lower=lower)
                x0[:] = spsolve(self.U1, b-self.L1@x0, permc_spec="NATURAL")


class JacobiSmoother():
    def __init__(self, A, isDDof=None):
        if isDDof is not None:
            # 处理 D 氏 自由度条件
            gdof = len(isDDof)
            bdIdx = np.zeros(gdof, dtype=np.int_)
            bdIdx[isDDof] = 1 
            Tbd = spdiags(bdIdx, 0, gdof, gdof)
            T = spdiags(1-bdIdx, 0, gdof, gdof)
            A = T@A@T + Tbd

        self.D = A.diagonal() 
        self.L = tril(A, k=-1).tocsr()
        self.U = triu(A, k=1).tocsr()

    def smooth(self, b, maxit=100):
        r = b.copy()
        for i in range(maxit):
            r[:] = b - self.L@r - self.U@r
            r /= self.D
        return r



class MG():
    def __init__(self, A, b, P, R=None, c=None, options=None):
        self.A = A
        self.b = b
        self.P = P
        if c:
            self.c = c
            print('c', c)


    def options(
            self,
            method='mean',
            maxrefine=3,
            maxcoarsen=0,
            theta=1.0,
            maxsize=1e-2,
            minsize=1e-12,
            data=None,
            HB=True,
            imatrix=False,
            disp=True
            ):

        options = {
                'method': method,
                'maxrefine': maxrefine,
                'maxcoarsen': maxcoarsen,
                'theta': theta,
                'maxsize': maxsize,
                'minsize': minsize,
                'data': data,
                'HB': HB,
                'imatrix': imatrix,
                'disp': disp
            }
        return options

    def pre_smoothing(self, A, b, x0):
        GS = GaussSeidelSmoother(A)
        GS.smooth(b, x0)

        return x0
 

    def solve(self):
        A = self.A
        b = self.b
        P = self.P
        n = len(P) # 获得加密层数
        x0 = np.zeros((A.shape[0],), dtype=np.float64)
        for i in range(n-1, -1, -1):
            P1 = P[i]
            R1 = self.c*P1.T
            ## 前磨光
            x0 = pre_smoothing(A, b, x0)
            r = b - A@x0
            print('P1', A.shape[0], P1.shape)
            print('R1', R1.shape)
            rH = R1@r
            print('x0', x0.shape, x0)

        print('P', len(P))

        return A
        
