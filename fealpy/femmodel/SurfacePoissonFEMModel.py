import numpy as np

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from ..quadrature  import TriangleQuadrature
from ..functionspace.surface_lagrange_fem_space import SurfaceLagrangeFiniteElementSpace
from ..solver import solve
from ..boundarycondition import DirichletBC

class SurfacePoissonFEMModel(object):
    def __init__(self, mesh, surface, model, integrator=None, p=1, p0=None):
        """
        """
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, surface, p=p, p0=p0) 
        self.mesh = self.V.mesh
        self.surface = surface
        self.model = model
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(model.solution)
        if integrator is None:
            self.integrator = TriangleQuadrature(p+1)
        if type(integrator) is int:
            self.integrator = TriangleQuadrature(integrator)
        else:
            self.integrator = integrator 
        self.area = self.V.mesh.area(integrator)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p) 
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.V.mesh.area(self.integrator)

    def get_left_matrix(self):
        V = self.V
        mesh = self.mesh
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.dof.cell2dof
        area = self.area

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        gphi = V.grad_basis(bcs)
        A = np.einsum('i, ijkm, ijpm->jkp', ws, gphi, gphi)
        A *= area.reshape(-1, 1, 1)
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        return A

    def get_right_vector(self):
        """
        Compute the right hand side.
        """
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights
        pp = mesh.bc_to_point(bcs)
        fval = model.source(pp)
        phi = V.basis(bcs)
        bb = np.einsum('i, ij, ik->kj', ws, phi, fval)
        bb *= self.area.reshape(-1, 1)
        gdof = V.number_of_global_dofs()
        b = np.bincount(V.dof.cell2dof.flat, weights=bb.flat, minlength=gdof)
        b -= np.mean(b)
        return b

    def solve(self):
        uh = self.uh
        g0 = lambda p: 0 
        bc = DirichletBC(self.V, g0, self.is_boundary_dof)
        solve(self, uh, dirichlet=bc, solver='direct')

    def is_boundary_dof(self, p):
        isBdDof = np.zeros(p.shape[0], dtype=np.bool)
        isBdDof[0] = True
        return isBdDof

    def l2_error(self):
        uh = self.uh.copy()
        uI = self.uI 
        uh += uI[0] 
        return np.sqrt(np.sum((uh - uI)**2)/len(uI))

    def L2_error(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights 
        pp = mesh.bc_to_point(bcs)
        n, ps = mesh.normal(bcs)
        l = np.sqrt(np.sum(n**2, axis=-1))
        area = np.einsum('i, ij->j', ws, l)/2.0

        val0 = self.uh.value(bcs)
        val1 = model.solution(ps)
        e = np.einsum('i, ij->j', ws, (val1 - val0)**2)
        e *= area
        return np.sqrt(e.sum()) 


    def H1_error(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights 
        pp = mesh.bc_to_point(bcs)

        val0, ps, n= V.grad_value_on_surface(self.uh, bcs)
        val1 = model.gradient(ps)
        l = np.sqrt(np.sum(n**2, axis=-1))
        area = np.einsum('i, ij->j', ws, l)/2.0
        e = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, e)
        e *=self.area
        return np.sqrt(e.sum()) 
