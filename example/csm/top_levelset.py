import numpy as np

import matplotlib.pyplot as plt

from scipy import ndimage

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d


class TopLevelSet:

    def __init__(self, nelx: int = 60, nely: int = 30, E: float = 1.0, nu: float = 0.3):
        
        # Material Properties
        self._E = E
        self._nu = nu
        self.KE, self.KTr, self.lambda_, self.mu = self.materialInfo(E, nu)

        # Structure and Optimization
        self._nelx = nelx
        self._nely = nely
        self._struc = np.ones((nely, nelx))
        self._lsf = self.reinit(self._struc)

        self._shapeSens = np.zeros((nely, nelx))
        self._topSens = np.zeros((nely, nelx))



    def reinit(self, struc):
        strucFull = np.zeros((struc.shape[0] + 2, struc.shape[1] + 2))
        strucFull[1:-1, 1:-1] = struc

        dist_to_0 = ndimage.distance_transform_edt(strucFull)
        dist_to_1 = ndimage.distance_transform_edt(strucFull - 1)

        temp_1 = dist_to_1 - 0.5
        temp_2 = dist_to_0 - 0.5

        lsf = (~strucFull.astype(bool)).astype(int) * temp_1 - strucFull * temp_2

        return lsf

    def stiffnessMatrix(self, k):
        K = np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])

        return K

    def materialInfo(self, E, nu):
        lambda_ = E * nu / ((1 + nu) * (1 - nu))
        mu = E / (2 * (1 + nu))
        k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3 * nu/8,
                  -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3 * nu/8])

        KE = E / (1 - nu**2) * self.stiffnessMatrix(k)

        k = np.array([1/3, 1/4, -1/3, 1/4, -1/6, -1/4, 1/6, -1/4])
        KTr = E / (1 - nu) * self.stiffnessMatrix(k)

        return KE, KTr, lambda_, mu

    def FE(self, struc, KE):
        nely, nelx = struc.shape
        K = lil_matrix((2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1)))
        F = lil_matrix((2*(nelx+1)*(nely+1), 1))
        U = np.zeros(2*(nelx+1)*(nely+1))

        for elx in range(nelx):
            for ely in range(nely):
                n1 = (nely+1) * elx + ely
                n2 = (nely+1) * (elx+1) + ely
                edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3]

                K[np.ix_(edof, edof)] += max(struc[ely, elx], 0.0001) * KE

        F[2 * (round(nelx/2)+1) * (nely+1) - 1] = 1
        
        fixeddofs = list( range( 2*(nely+1)-2, 2*(nely+1) ) ) + list( range( 2*(nelx+1)*(nely+1)-2, 2*(nelx+1)*(nely+1) ) )
        alldofs = list( range( 2*(nely+1)*(nelx+1) ) )
        freedofs = list( set(alldofs) - set(fixeddofs) )

        U[freedofs] = spsolve(csc_matrix(K[np.ix_(freedofs, freedofs)]), F[freedofs])

        return U


    def optimize(self, Num: int = 200):
        self._objective = np.zeros(Num)

        for iterNum in range(Num):
            U = self.FE(self._struc, self.KE)

            for elx in range(self._nelx):
                for ely in range(self._nely):
                    n1 = (self._nely + 1) * elx + ely
                    n2 = (self._nely + 1) * (elx + 1) + ely
                    
                    Ue = U[np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3])]

                    self._shapeSens[ely, elx] = -max(self._struc[ely, elx], 0.0001) * Ue.T @ self.KE @ Ue
                    
                    coeff = np.pi/2 * (self.lambda_ + 2*self.mu) / self.mu / (self.lambda_ + self.mu)
                    UeT_KE_Ue = 4 * self.mu * Ue.T @ self.KE @ Ue
                    additional_term = (self.lambda_ - self.mu) * Ue.T @ self.KTr @ Ue

                    self._topSens[ely, elx] = self._struc[ely, elx] * coeff * UeT_KE_Ue * (UeT_KE_Ue + additional_term)


            self._objective[iterNum] = -np.sum(self._shapeSens)
            volCurr = np.sum(self._struc) / (self._nelx * self._nely)
            print(f'Iter: {iterNum+1}, Compliance.: {self._objective[iterNum]:.4f}, Volfrac.: {volCurr:.3f}')

            


tls = TopLevelSet()
print(tls.optimize(Num=1))
# print(test_reinit)
# test_u = tls.FE(tls._struc, tls.KE)
# print(test_u)
