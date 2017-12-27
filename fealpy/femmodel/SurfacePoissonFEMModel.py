import numpy as np

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from ..quadrature  import TriangleQuadrature
from ..functionspace.surface_lagrange_fem_space import SurfaceLagrangeFiniteElementSpace
from ..solver import solve
from ..boundarycondition import DirichletBC

class SurfacePoissonFEMModel(object):
    def __init__(self, mesh, surface, model, p=1, dtype=np.float):
        """
        """
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, surface, p, dtype=dtype) 
        self.surface = surface
        self.model = model
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(model.solution)
        self.area = self.V.mesh.area()
        self.dtype = dtype

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p, dtype=self.dtype) 
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.V.mesh.area()

    def get_left_matrix(self):
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells() 
        qf = TriangleQuadrature(4)#TODO: automatically choose numerical formula
        nQuad = qf.get_number_of_quad_points()
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        area = self.area 
        A = coo_matrix((gdof, gdof), dtype=self.dtype)
        for i in range(ldof):
            for j in range(i, ldof):
                val = np.zeros((NC,), dtype=self.dtype)
                for q in range(nQuad):
                    lambda_k, w_k = qf.get_gauss_point_and_weight(q)
                    gradphi = V.grad_basis(lambda_k)
                    val += np.sum(gradphi[:,i,:]*gradphi[:,j,:], axis=1)*w_k
                A += coo_matrix((val*area, (cell2dof[:,i], cell2dof[:,j])), shape=(gdof, gdof))
                if j != i:
                    A += coo_matrix((val*area, (cell2dof[:,j], cell2dof[:,i])), shape=(gdof, gdof))
        return A.tocsr()


    def get_right_vector(self):
        """
        Compute the right hand side.
        """
        V = self.V
        mesh = V.mesh
        model = self.model
        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(4)#TODO:
        nQuad = qf.get_number_of_quad_points()
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        bb = np.zeros((NC,ldof),dtype=self.dtype)
        area = self.area
        # Compute the integral average of the source `f`
        fh = np.zeros(NC, dtype=self.dtype)
        for i in range(nQuad):
            bc,w = qf.get_gauss_point_and_weight(i)
            ps = mesh.bc_to_point(bc)
            ps, _= self.surface.project(ps)
            fval = model.source(ps) 
            fh += fval*w
        fh *= area
        fh = np.sum(fh)/np.sum(area)

        for i in range(nQuad):
            bc,w = qf.get_gauss_point_and_weight(i)
            ps = mesh.bc_to_point(bc)
            ps, _ = self.surface.project(ps)
            fval = model.source(ps) - fh 
            phi = V.basis(bc)
            bb += w*phi*fval.reshape(-1, 1)

        bb *= area.reshape(-1, 1)
        b = np.zeros((gdof,),dtype=self.dtype)
        np.add.at(b, cell2dof.flatten(), bb.flatten())
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
        NC = mesh.number_of_cells()    
        qf = TriangleQuadrature(8)
        nQuad = qf.get_number_of_quad_points()
        e = np.zeros((NC,), dtype=self.dtype)
        area = np.zeros((NC,), dtype=self.dtype)
        for i in range(nQuad):
            bc, w = qf.get_gauss_point_and_weight(i)
            uhval = self.uh.value(bc)
            ps = mesh.bc_to_point(bc)                                      

            J0, _, _, _ = mesh.jacobi(bc)
            J1 = self.surface.jacobi(ps)
            J = np.einsum('ijk, imk->imj', J1, J0)
            n = np.cross(J[:, 0, :], J[:, 1, :], axis=1)
            area += np.sqrt(np.sum(n**2, axis=1))*w

            uval = model.solution(ps)
            e += w*(uhval - uval)*(uhval - uval)
        e *= area/2.0
        return np.sqrt(e.sum())


    def H1_error(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        p = v.p
        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(8)
        nQuad = qf.get_number_of_quad_points()
        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        e = np.zeros((NC,), dtype=self.dtype)
        for i in range(nQuad):
            bc, w = qf.get_gauss_point_and_weight(i)
            gval = self.uh.grad_value(bc)
            p = mesh.bc_to_point(bc)
            p, _ = self.surface.project(p)
            val = model.gradient(p)
            e += w*((gval - val)*(gval - val)).sum(axis=1)
        e *= mesh.area()
        return np.sqrt(e.sum())

