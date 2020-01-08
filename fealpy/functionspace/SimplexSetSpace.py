import numpy as np

from .femdof import multi_index_matrix2d

class SimplexSetSpace():
    def __init__(self, TD, ftype=np.float, itype=np.int):
        self.TD = TD
        self.multi_index_matrix = multi_index_matrix2d
        self.ftype = ftype
        self.itype = itype

    def basis(self, bc, p=1):
        TD = self.TD
        multiIndex = self.multi_index_matrix(p)
        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def value(self, data, bc, p=1):
        phi = self.basis(bc, p=p)
        s1 = '...j, ij->...i'
        val = np.einsum(s1, phi, data)
        return val
