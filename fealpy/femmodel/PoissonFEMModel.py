import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace  
from ..quadrature  import TriangleQuadrature
from ..solver import solve
from ..boundarycondition import DirichletBC

class PoissonFEMModel(object):
    def __init__(self, mesh,  model, p=1):
        self.V = LagrangeFiniteElementSpace(mesh, p) 
        self.mesh = mesh 
        self.model = model
        self.uh = self.V.function()
        self.uI = self.V.interpolation(model.solution)
        self.area = self.mesh.area()

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = LagrangeFiniteElementSpace(mesh, p) 
        self.uh = self.V.function()
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.mesh.area()

    
    def get_left_matrix(self):
        V = self.V
        mesh = self.mesh
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.dof.cell2dof
        area = self.area

        qf = TriangleQuadrature(6)
        bcs, ws = qf.quadpts, qf.weights
        gphi = V.grad_basis(bcs)
        A = np.einsum('i, ijkm, ijpm->jkp', ws, gphi, gphi)
        A *= area.reshape(-1, 1, 1)
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def get_right_vector(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = TriangleQuadrature(6)
        bcs, ws = qf.quadpts, qf.weights
        pp = mesh.bc_to_point(bcs)
        fval = model.source(pp)
        phi = V.basis(bcs)
        bb = np.einsum('i, ij, ik->kj', ws, phi,fval)

        bb *= self.area.reshape(-1, 1)
        gdof = V.number_of_global_dofs()
        b = np.bincount(V.dof.cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b


    def solve(self):
        bc = DirichletBC(self.V, self.model.dirichlet)
        solve(self, self.uh, dirichlet=bc, solver='direct')

    def l2_error(self):
        uh = self.uh
        uI = self.uI
        return np.sqrt(np.sum((uh - uI)**2)/len(uI))  

    def L2_error(self, order=6):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = TriangleQuadrature(order)
        bcs, ws = qf.quadpts, qf.weights 
        pp = mesh.bc_to_point(bcs)

        val0 = self.uh.value(bcs)
        val1 = model.solution(pp)
        e = np.einsum('i, ij->j', ws, (val1 - val0)**2)
        e *=self.area
        return np.sqrt(e.sum()) 

    
    def H1_error(self, order=6):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = TriangleQuadrature(order)
        bcs, ws = qf.quadpts, qf.weights 
        pp = mesh.bc_to_point(bcs)

        val0 = self.uh.grad_value(bcs)
        val1 = model.gradient(pp)
        e = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, e)
        e *=self.area
        return np.sqrt(e.sum()) 

	


