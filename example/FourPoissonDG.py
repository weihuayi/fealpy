import numpy as np

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat
from scipy.sparse.linalg import spsolve

from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.quadrature import GaussLegendreQuadrature
from fealpy.decorator import cartesian


class DGMethod():
    def __init__(self, pde, mesh, p):
        self.uv=0
        self.space = ScaledMonomialSpace2d(mesh, p)
        self.mesh = mesh
        self.pde = pde 
        self.p = p
        self.I = ~mesh.ds.boundary_edge_flag()
        self.B = mesh.ds.boundary_edge_flag()
    
    def get_left_matrix(self,beta,alpha):
        A = self.space.stiff_matrix()
        J = self.space.penalty_matrix(index=self.I)
        Q = self.space.normal_grad_penalty_matrix(index=self.I)
        S0 = self.space.flux_matrix(index=self.I)
        S1 = self.space.flux_matrix()

        A11 = A-S0-S1.T+alpha*J+beta*Q
        A12 = -self.space.mass_matrix()

        A22 = A11.T-A12
        A21 = alpha*self.space.penalty_matrix() 
        AD = bmat([[A11, A12], [A21, A22]], format='csr')
        return AD

    def get_right_vector(self):
        F11 = self.space.edge_source_vector(self.pde.gradient, index=self.B,
                hpower=0)
        F12 = -self.space.edge_normal_source_vector(self.pde.dirichlet,
                index=self.B)
        F21 = self.space.edge_source_vector(self.pde.dirichlet, index=self.B)
        F22 = self.space.source_vector0(self.pde.source)
        F = np.r_[F11+F12, F21+F22]
        return F

    def solve(self,beta,alpha):
        gdof = self.space.number_of_global_dofs(p=self.p)
        AD = self.get_left_matrix(beta,alpha)
        b = self.get_right_vector()
        self.uh = self.space.function()
        self.uv = spsolve(AD, b)
        self.uh[:] = self.uv[:gdof]
        ls = {'A':AD, 'b':b, 'solution':self.uh.copy()}

        return ls # return the linear system


