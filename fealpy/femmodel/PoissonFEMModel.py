import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.tools import function_space

from ..quadrature  import TriangleQuadrature
from ..solver import solve
from ..boundarycondition import DirichletBC

class PoissonFEMModel(object):
    def __init__(self, mesh,  model, p=1, dtype=np.float):
        self.V = function_sapce(mesh,'Lagrange', p, dtype=dtype)
        self.mesh = mesh 
        self.model = model
        self.uh = self.V.FiniteElementFunction()
        self.uI = self.V.interpolation(model.solution)

        self.dtype = dtype

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.function_space(mesh,femtype,p,dtype=self.dtype)
        self.uh = self.V.FiniteElementFunction()
        self.area = self.V.mesh.area()

    
    def get_left_matrix(self):
        V = self.V
        mesh = self.mesh
        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(4)
        nQuad = qf.get_number_of_quad_points()
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        area = self.area
        A = coo_matrix((gdof, gdof), dtype=self.dtype)
        for i in range(ldof):
            for j in range(i,ldof):
                val = np.zeros((NC,), dtype=self.dtype)
                for q in range(nQuad):
                    bc, w = qf.get_gauss_point_and_weight(q)
                    gphi = grad_basis_einsum(bc)
                    val += np.sum(gphi[:,i,:]*gphi[:,j,:], axis=1)*w
                A += coo_matrix((val*area, (cell2dof[:,i], cell2dof[:,j])), shape=(gdof, gdof))
                if j != i:
                    A += coo_matrix((val*area, (cell2dof[:,j], cell2dof[:,i])), shape=(gdof, gdof))
        return A.tocsr()


    def get_right_vector(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(4)
        nQuad = qf.get_number_of_quad_points()
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        bb = np.zeros((NC, ldof), dtype=self.dtype)
        area = mesh.area()
        for i in range(nQuad):
            bc, w = qf.get_gauss_point_and_weight(i)
            p = mesh.bc_to_point(bc)
            fval = model.source(p)
            phi = V.basis_einsum(bc)
            for j in range(ldof):
                bb[:, j] += fval*phi[j]*w_k
        
        bb *= area.reshape(-1, 1)
        cell2dof = V.cell_to_dof()
        b = np.zeros((gdof,), dtype=self.dtype)
        np.add.at(b, cell2dof.flatten(), bb.flatten())
        return b


    def solve(self):
        uh = self.uh
        bc = DirichletBC(V, model.dirichlet, model.is_boundary)
        solve(self, uh, dirichlet=bc, solver='direct')

    
    def is_boundary_dof(self,p):
        isBdDof = np.zeros(p.shape[0], dtype=np.bool)
        isBdDof[0] = True
        return isBdDof

    def l2_error(self):
        uh = self.uh.copy()
        uI = self.uI
        uh += uI[0]
        return np.sqrt(np.sum((uh - uI)**2)/len(uI))  
    


    def L2_error(self, order=4):
        V = uh.V
        mesh = V.mesh
        model = self.model
        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(order)
        nQuad = qf.get_number_of_quad_points()
        
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        
        e = np.zeros((NC,), dtype=dtype)
        for i in range(nQuad):
            bc, w = qf.get_gauss_point_and_weight(i)
            uhval = uh.value(bc)
            p = mesh.bc_to_point(bc)
            uval = model.source(p)
            if len(uval.shape) == 1:
                e += w_k*(uhval - uval)*(uhval - uval)
            else:
                e += w*((uhval - uval)*(uhval - uval)).sum(axis=1)
        e *= mesh.area()
        return np.sqrt(e.sum()) 

    
    def H1_error(self, order=4):
        V = uh.V
        mesh = V.mesh
        model = self.model
        
        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(order)
        nQuad = qf.get_number_of_quad_points()
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        e = np.zeros((NC,), dtype=dtype)
        for i in range(nQuad):
            bc, w = qf.get_gauss_point_and_weight(i)
            gval = uh.grad_value(bc)
            p = mesh.bc_to_point(bc)
            val = model.gradient(p)
            e += w*((gval - val)*(gval - val)).sum(axis=1)
        e *= mesh.area()
        return np.sqrt(e.sum())

	


