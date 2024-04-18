
import numpy as np
from functools import partial
from scipy.sparse import csr_matrix

class ScalarBiharmonicIntegrator:

    def __init__(self, q=None):
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], out=None):
        """
        @brief 计算三角形网格上的单元 hessian 矩阵
        """
        mesh = space.mesh

        p = space.p
        q = self.q if self.q is not None else p+4 

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        cm = mesh.entity_measure(index=index)

        hphi = space.hess_basis(bcs)
        A = np.einsum('c, q, qclij, qcmij->clm', cm, ws, hphi, hphi)
        if out is None:
            return A
        else:
            out += A

