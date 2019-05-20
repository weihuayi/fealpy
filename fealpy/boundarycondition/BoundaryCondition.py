import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

class DirichletBC:
    def __init__(self, V, g0, is_dirichlet_dof=None):
        self.V = V
        self.g0 = g0

        if is_dirichlet_dof == None:
            isBdDof = V.boundary_dof()
        else:
            ipoints = V.interpolation_points()
            isBdDof = is_dirichlet_dof(ipoints)

        self.isBdDof = isBdDof

    def apply(self, A, b):
        """ Modify matrix A and b
        """
        g0 = self.g0
        V = self.V
        isBdDof = self.isBdDof

        gdof = V.number_of_global_dofs()
        x = np.zeros((gdof,), dtype=np.float)
        ipoints = V.interpolation_points()
        # the length of ipoints and isBdDof maybe different
        idx, = np.nonzero(isBdDof)
        x[isBdDof] = g0(ipoints[idx])
        b -= A@x
        bdIdx = np.zeros(gdof, dtype=np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        A = T@A@T + Tbd

        b[isBdDof] = x[isBdDof] 
        return A, b

    def apply_on_matrix(self, A):

        V = self.V
        isBdDof = self.isBdDof
        gdof = V.number_of_global_dofs()

        bdIdx = np.zeros((A.shape[0], ), np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd

        return A

    def apply_on_vector(self, b, A):
        
        g0 = self.g0
        V = self.V
        isBdDof = self.isBdDof

        gdof = V.number_of_global_dofs()
        x = np.zeros((gdof,), dtype=np.float)

        ipoints = V.interpolation_points()
        x[isBdDof] = g0(ipoints[isBdDof,:])
        b -= A@x

        b[isBdDof] = x[isBdDof] 

        return b



        


