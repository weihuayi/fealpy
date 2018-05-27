import numpy as np

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from ..quadrature  import TriangleQuadrature
from ..functionspace.surface_lagrange_fem_space import SurfaceLagrangeFiniteElementSpace
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..femmodel import doperator 
from .SurfaceIntegralAlg import SurfaceIntegralAlg

class SurfacePoissonFEMModel(object):
    def __init__(self, mesh, pde, p, k, p0=None):
        """
        """
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, pde.surface, p=p, p0=p0) 
        self.mesh = self.V.mesh
        self.surface = pde.surface
        self.pde = pde
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(pde.solution)
        self.integrator = self.mesh.integrator(k)
        self.area = self.V.mesh.area(self.integrator)
        self.error = SurfaceIntegralAlg(self.integrator, self.mesh, self.area)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p) 
        self.mesh = self.V.mesh
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(self.pde.solution)
        self.area = self.V.mesh.area(self.integrator)
        self.error.area = self.area 
        self.error.mesh = self.mesh

    def recover_estimate(self):
        if self.V.p > 1:
            raise ValueError('This method only work for p=1!')

        V = self.V
        mesh = V.mesh.mesh

        p2c = mesh.ds.node_to_cell()
        inva = 1/mesh.area()
        asum = p2c@inva

        bc = np.array([1/3]*3, dtype=np.float)
        guh = self.uh.grad_value(bc)

        VV = VectorLagrangeFiniteElementSpace(mesh, p=1)
        rguh = VV.function()
        rguh[:] = np.asarray(p2c@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights

        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.area
        return np.sqrt(e)

    def get_left_matrix(self):
        return doperator.stiff_matrix(self.V, self.integrator, self.area)

    def get_right_vector(self):
        b = doperator.source_vector(self.pde.source, self.V, self.integrator,
                self.area, self.surface)
        b -= np.mean(b)
        return b 

    def solve(self):
        uh = self.uh
        u = self.pde.solution
        bc = DirichletBC(self.V, u, self.is_boundary_dof)
        solve(self, uh, dirichlet=bc, solver='direct')

    def is_boundary_dof(self, p):
        isBdDof = np.zeros(p.shape[0], dtype=np.bool)
        isBdDof[0] = True
        return isBdDof

    def l2_error(self):
        u = self.pde.solution
        uh = self.uh
        return self.error.l2_error(u, uh)

    def L2_error(self):
        u = self.pde.solution
        uh = self.uh.value
        return self.error.L2_error(u, uh)

    def H1_semi_error(self):
        gu = self.pde.gradient
        guh = self.uh.grad_value
        return self.error.H1_semi_error(gu, guh)

