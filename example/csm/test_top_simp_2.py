import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

class TopSimp_2:
    '''
    One load short cantilever
    '''
    def __init__(self, nelx: int = 32, nely: int = 20, volfrac: float = 0.4, 
                 E: float = 1.0, nu: float = 0.3, penal: float = 3.0, rmin: float = 1.2):
        self._nelx = nelx
        self._nely = nely
        self._volfrac = volfrac
        self._E = E
        self._nu = nu
        self._penal = penal
        self._rmin = rmin
        self._x = np.full((nely, nelx), volfrac)

    def lk(self):
        E, nu = self._E, self._nu
        k = [1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8]
        KE = E / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
        
        return KE
    

    def FE(self, nelx, nely, penal, x):
        KE = self.lk()
        K = lil_matrix( ( 2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1) ) )
        F = lil_matrix( ( 2*(nelx+1)*(nely+1), 1) )
        U = np.zeros( 2*(nely+1)*(nelx+1) )
        
        for elx in range(nelx):
            for ely in range(nely):
                n1 = (nely+1) * elx + ely
                n2 = (nely+1) * (elx+1) + ely
                edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
                print("edof:", edof)

                K[np.ix_(edof, edof)] += x[ely, elx] ** penal * KE

        # print("K:\n", K.toarray().round(4))
        F[2*(nelx+1)*(nely+1) - 1] = -1
        fixeddofs = np.arange(2*(nely+1))
        alldofs = np.arange(2 * (nely+1) * (nelx+1))
        freedofs = np.setdiff1d(alldofs, fixeddofs)
        
        U[freedofs] = spsolve(csc_matrix(K[np.ix_(freedofs, freedofs)]), F[freedofs])
        U[fixeddofs] = 0
        
        return U


    def check(self, nelx, nely, rmin, x, dc):
        assert np.all(dc <= 0), "dc should be non-positive (usually negative)"

        dcn = np.zeros((nely, nelx))

        for i in range(nelx):
            for j in range(nely):
                sum_val = 0.0  
                for k in range( max( i - int(rmin), 0 ), min( i + int(rmin) + 1, nelx ) ):
                    for l in range( max( j - int(rmin), 0 ), min( j + int(rmin) + 1, nely ) ):
                        fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                        sum_val += max(0, fac)
                        dcn[j, i] += max(0, fac) * x[l, k] * dc[l, k]
                dcn[j, i] /= (x[j, i] * sum_val)

        return dcn


    def OC(self, nelx, nely, volfrac, x, dc):
        assert np.all(dc <= 0), "dc should be non-positive (usually negative)"

        l1, l2 = 0, 1e5
        move = 0.2  
        xnew = np.copy(x)  

        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            tmp0 = x - move
            tmp1 = x + move
            tmp2 = x * np.sqrt(-dc / lmid)
            tmp3 = np.minimum(tmp1, tmp2)
            tmp4 = np.minimum(1, tmp3) 
            tmp5 = np.maximum(tmp0, tmp4)
            xnew = np.maximum(0.001, tmp5)
            if np.sum(xnew) - volfrac * nelx * nely > 0:
                l1 = lmid
            else:
                l2 = lmid

        return xnew

    def optimize(self):
        nelx, nely, rmin, penal, x, volfrac = self._nelx, self._nely, self._rmin, self._penal, self._x, self._volfrac

        loop = 0
        change = 1.0

        while change > 0.01:
            loop += 1
            xold = np.copy(x)
            U = self.FE(nelx, nely, penal, x)
            # if loop == 1 or loop == 2:
                # print(f"U for loop {loop}:", U.shape, U)
            KE = self.lk()
            c = 0 
            dc = np.zeros((nely, nelx)) 

            for elx in range(nelx):
                for ely in range(nely):
                    n1 = (nely+1) * elx + ely
                    n2 = (nely+1) * (elx+1) + ely
                    edof = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1, 2*n2 + 2, 2*n2 + 3, 2*n1 + 2, 2*n1 + 3])
                    Ue = U[edof]
                    c += x[ely, elx]**penal * Ue.T @ KE @ Ue 
                    dc[ely, elx] = -penal * x[ely, elx]**(penal - 1) * Ue.T @ KE @ Ue

            dc = self.check(nelx, nely, rmin, x, dc)
            x = self.OC(nelx, nely, volfrac, x, dc)

            change = np.max(np.abs(x - xold))
            # print(f' Iter.: {loop:4d} Objective.: {c:10.4f} Volfrac.: {np.sum(x)/(nelx*nely):6.3f} change.: {change:6.3f}')
            # 
            # plt.imshow(-x, cmap='gray')
            # plt.axis('off')
            # plt.axis('equal')
            # plt.draw()
            # plt.pause(1e-5)
            
        # plt.ioff()
        # plt.show()

# 创建一个TopSimp对象实例
tsp = TopSimp_2(nelx=6, nely=5)
print(tsp.optimize())
