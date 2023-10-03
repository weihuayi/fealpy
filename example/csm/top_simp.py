import numpy as np

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

class TopSimp:
    def __init__(self, nelx: int = 60, nely: int = 20, volfrac: float = 0.5, 
                E: float = 1.0, nu: float = 0.3, penal: float = 3.0):
        self._nelx = nelx
        self._nely = nely
        self._volfrac = volfrac
        self._E = E
        self._nu = nu
        self._penal = penal
        self._x = np.full((nely, nelx), volfrac)
        self._loop = 0
        self._change = 1.0

    def lk(self) -> np.ndarray:
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
    

    def FE(self, x: np.ndarray) -> np.ndarray:
        nelx, nely, penal = self._nelx, self._nely, self._penal  
        KE = self.lk()
        K = lil_matrix( ( 2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1) ) )
        F = lil_matrix( ( 2*(nelx+1)*(nely+1), 1) )
        U = np.zeros( 2*(nely+1)*(nelx+1) )
        
        for elx in range(nelx):
            for ely in range(nely):
                n1 = (nely+1) * elx + ely
                n2 = (nely+1) * (elx+1) + ely
                edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])

                K[np.ix_(edof, edof)] += x[ely, elx] ** penal * KE    

        F[1] = -1
        fixeddofs = np.union1d(np.arange(0, 2*(nely+1), 2), np.array([2*(nelx+1)*(nely+1) - 1]))
        alldofs = np.arange(2 * (nely+1) * (nelx+1))
        freedofs = np.setdiff1d(alldofs, fixeddofs)
        
        U[freedofs] = spsolve(csc_matrix(K[np.ix_(freedofs, freedofs)]), F[freedofs])
        U[fixeddofs] = 0
        
        return U

    def optimize(self):
        while self._change > 0.01:
            self._loop += 1
            xold = np.copy(self._x)
            U = self.FE(self._x)

# 创建一个TopSimp对象实例
top_simp = TopSimp()
