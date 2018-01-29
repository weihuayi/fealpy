import numpy as np
from itertools import compress
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..vemmodel import doperator 
from ..functionspace import FunctionNorm

from timeit import default_timer as timer

class PoissonInterfaceVEMModel():
    def __init__(self, model, mesh, p=1):
        """
        Parameters
        ----------
        
        See Also
        --------

        Notes
        -----
        """
        self.V =VirtualElementSpace2d(mesh, p) 

        barycenter = self.V.smspace.barycenter
        isIntCell = model.interface(barycenter) < 0
        point = mesh.point
        N = mesh.number_of_points()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        edge = mesh.ds.edge 
        edge2cell = mesh.ds.edge_to_cell()
        isInterfaceEdge = (isIntCell[edge2cell[:, 0]] != isIntCell[edge2cell[:, 1]])
        isInterfacePoint = np.zeros(N, dtype=np.bool)
        isInterfacePoint[edge[isInterfaceEdge]] = True
        self.interfaceEdge = edge[isInterfaceEdge]

        self.wh = self.V.function()
        self.wh[isInterfacePoint] = self.model.func_jump(point[isInterfacePoint])

        isExtPoint = np.zeros(N, dtype=np.bool)
        isExtPoint[cell[np.repeat(~isIntCell, NV)]] = True
        self.uI0 = self.V.function() 
        self.uI0[isExtPoint] = model.solution_plus(point[isExtPoint])
        isIntPoint = np.zeros(N, dtype=np.bool)
        isIntPoint[cell[np.repeat(isIntCell, NV)]] = True
        self.uI1 = self.V.function()
        self.uI1[isIntPoint] = model.solution_minus(point[isIntPoint])

        self.isIntCell = isIntCell
        self.model = model  
        self.uh = self.V.function() 
        self.area = self.V.smspace.area 

        self.H = doperator.matrix_H(self.V)
        self.D = doperator.matrix_D(self.V, self.H)
        self.B = doperator.matrix_B(self.V)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = VirtualElementSpace2d(mesh, p) 
        self.uh = self.V.function() 
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.V.smspace.area

        self.H = doperator.matrix_H(self.V)
        self.D = doperator.matrix_D(self.V, self.H)
        self.B = doperator.matrix_B(self.V)

    def project_to_smspace(self, uh=None):
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        ldof = V.smspace.number_of_local_dofs()

        S = self.V.smspace.function()
        if uh is None:
            uh = self.uh
        idx = np.repeat(range(NC), NV)
        for i in range(3):
            S[i::ldof] = np.bincount(idx, weights=self.B[i, :]*uh[cell], minlength=NC)
        return S


    def get_left_matrix(self):
        return self.get_stiff_matrix() 

    def get_right_vector(self):
        area = self.area
        V = self.V
        f = self.model.source 
        b0 =  doperator.source_vector(f, V, area)
        b1 = self.get_flux_jump_vector()
        return b0 - b1 + self.AI@self.wh 

    def get_flux_jump_vector(self):
        qf = GaussLegendreQuadrture(3)
        bcs, ws = qf.quadpts, qf.weights 
        point = self.mesh.point
        iedge = self.interfaceEdge
        ps = np.einsum('ij, kjm->ikm', bcs, point[iedge])
        val = self.model.flux_jump(ps)
        bb = np.einsum('i, ij, ik->kj', ws, bcs, val)

        l = np.sqrt(np.sum((point[iedge[:, 0], :] - point[iedge[:, 1], :])**2, axis=1))

        bb *= l.reshape(-1, 1)
        gdof = self.V.number_of_global_dofs()
        b = np.bincount(iedge.flat, weights=bb.flat, minlength=gdof)
        return b

    def get_stiff_matrix(self):
        B = self.B
        D = self.D

        cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
        NC = len(cell2dofLocation) - 1
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        BB = np.hsplit(B, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])
        tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])

        barycenter = self.V.smspace.barycenter 
        k = self.model.diffusion_coefficient(barycenter)
        f1 = lambda x: (x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
        K = list(map(f1, zip(DD, BB, k)))

        f2 = lambda x: np.repeat(x, x.shape[0]) 
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()


        I = np.concatenate(list(map(f2, cd)))
        J = np.concatenate(list(map(f3, cd)))
        val = np.concatenate(list(map(f4, K)))
        gdof = V.number_of_global_dofs()
        A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)

        I = np.concatenate(list(map(f2, compress(cd, self.isIntCell))))
        J = np.concatenate(list(map(f3, compress(cd, self.isIntCell))))
        val = np.concatenate(list(map(f4, compress(K, self.isIntCell))))
        AI = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
        self.AI = AI
        return A

    def solve(self):
        uh = self.uh
        bc = DirichletBC(self.V, self.model.dirichlet)
        self.A, b = solve(self, uh, dirichlet=bc, solver='direct')
        self.S = self.project_to_smspace(uh)
    
    def l2_error(self):
        u = self.model.solution
        uh = self.uh
        return self.error.l2_error(u, uh)

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self, quadtree):
        u = self.model.solution
        uh = self.S.value
        return self.error.L2_error(u, uh, quadtree, barycenter=False)

    def H1_semi_error(self, quadtree):
        gu = self.model.gradient
        guh = self.S.grad_value
        return self.error.L2_error(gu, guh, quadtree, barycenter=False)

