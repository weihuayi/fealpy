import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from .function import Function
from ..quadrature import FEMeshIntegralAlg
from timeit import default_timer as timer




class PrismFiniteElementSpace():

    def __init__(self, mesh, p=1, q=None):
        self.mesh = mesh
        self.p = p

        if q is None:
            self.integrator = mesh.integrator(p+1)
        else:
            self.integrator = mesh.integrator(q)

        self.integralalg = FEMeshIntegralAlg(self.integrator, self.mesh)


    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.dpoints

    def cell_to_dof(self):
        return self.dof.cell2dof

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def lagranian_basis(self, bc, TD):
        p = self.p   # the degree of polynomial basis function
        multiIndex = self.dof.multi_index_matrix(TD)
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

    def basis(self, bc):
        bc0 = bc[0]
        bc1 = bc[1]
        phi0 = self.lagranian_basis(bc0, 2)
        phi1 = self.lagranian_basis(bc1, 1)
        phi = np.eisum('', phi0, phi1)
        return phi

    def grad_basis(self, bc, cellidx=None):
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)
