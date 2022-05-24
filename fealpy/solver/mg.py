import numpy as np
from numpy.linalg import norm
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
    def __init__(self, Ah, bh, P, R=None, c=None, options=None):
        self.P = P
        self.bh = bh
        if c:
            self.c = c
        if c is None:
            self.c = 1
        
        self.A = []
        self.A.append(Ah)
        P = self.P
        n = len(P) # 获得加密层数
        for i in range(n):
            P1 = P[n-1-i]
            R1 = self.c*P1.T
            Ah = R1@Ah@P1
            self.A.append(Ah)


    def options(
            self,
            method='mean',
            maxrefine=3,
            maxcoarsen=0,
            theta=1.0,
            maxsize=1e-2,
            minsize=1e-12,
            tol = 1e-9,
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
                'tol': tol,
                'data': data,
                'HB': HB,
                'imatrix': imatrix,
                'disp': disp
            }
        return options

    def pre_smoothing(self, A, b, x0):

        r = []
        e = []
        r.append(b)
        n = len(A)
        if n == 1:
            raise Exception("网格已经足够粗")
        
        GS = GaussSeidelSmoother(A[0])
        GS.smooth(b, x0)
        e.append(x0)
        rh = b - A[0]@x0
        rH = self.c*self.P[n-2].T@rh
        r.append(rH) 
        e0 = np.zeros(rH.shape[0], dtype=np.float64)
        if n == 2:
            return r, e

        for i in range(2,n):
            GS = GaussSeidelSmoother(A[i-1])
            GS.smooth(rH, e0)
            e.append(e0)
            rh = rH - A[i-1]@e0
            rH = self.c*self.P[n-i-1].T@rh
            r.append(rH)
            e0 = np.zeros(rH.shape[0], dtype=np.float64)           
#            e0 = self.c*self.P[n-i-1].T@e0

        return r, e


    def post_smoothing(self, A, b, x0):
        n = len(A)
        eh = x0[-1]
        if n == 1:
            raise Exception("网格已经足够粗")
        for i in range(n-1):
            eh = x0[n-i-2] + self.P[i]@eh
            GS = GaussSeidelSmoother(A[n-i-2])
            GS.smooth(b[n-i-2], eh, lower=False)
 
        return eh
 

    def solve(self, x0):
        A = self.A
        bh = self.bh
        ## 前磨光
        r, x = self.pre_smoothing(A, bh, x0)
        
        ## 最粗网格求解
        x0 = spsolve(A[-1], r[-1])
        x.append(x0)

        ## 后磨光
        x0 = self.post_smoothing(A, r, x)

        return x0

    def v_cycle(self):
        
        count = 0
        ru = 1
        tol = 1e-9
        u0 = np.zeros(self.bh.shape[0], dtype=np.float64)

        while ru > tol and count < 100:
            uh = self.solve(u0)
            if norm(self.bh) == 0:
                ru = norm(self.bh - self.A[0]@uh)
            else:
                ru = norm(self.bh - self.A[0]@uh)/norm(self.bh)

            u0[:] = uh[:]
            count += 1

            print('ru', ru)
        print('count', count)

        return uh
        
