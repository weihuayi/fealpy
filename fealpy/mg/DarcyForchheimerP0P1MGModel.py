import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy.linalg import norm
from scipy.sparse.linalg import cg, inv, spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..mesh import TriangleMesh

class DarcyForchheimerP0P1MGModel:

    def __init__(self, pde, mesh, n):

        self.integrator = mesh.integrator(3)

        self.pde = pde
        self.uspaces = []
        self.pspaces = []
        self.IMatrix = []
        self.A = []
        self.b = []

        mesh0 = TriangleMesh(mesh.node, mesh.ds.cell)
        uspace = VectorLagrangeFiniteElementSpace(mesh0, p=0, spacetype='D')
        self.uspaces.append(uspace)

        pspace = LagrangeFiniteElementSpace(mesh0, p=1, spacetype='C')
        self.pspaces.append(pspace)

        for i in range(n):
            I0, I1 = mesh.uniform_refine(returnim=True)
            self.IMatrix.append((I0[0], I1[0]))
            mesh0 = TriangleMesh(mesh.node, mesh.ds.cell)
            uspace = VectorLagrangeFiniteElementSpace(mesh0, p=0, spacetype='D')
            self.uspaces.append(uspace)
            pspace = LagrangeFiniteElementSpace(mesh0, p=1, spacetype='C')
            self.pspaces.append(pspace)

        self.uh = self.uspaces[-1].function()
        self.ph = self.pspaces[-1].function()

        self.uI = self.uspaces[-1].interpolation(pde.velocity)
        self.pI = self.pspaces[-1].interpolation(pde.pressure)

        self.nlevel = n + 1

        self.A = self.get_linear_stiff_matrix()
        self.b = self.get_right_vector()
        
    def get_linear_stiff_matrix(self, level):
        
        mesh = self.pspaces[level].mesh
        pde = self.pde
        mu = pde.mu
        rho = pde.rho

        bc = np.array([1/3,1/3,1/3], dtype=mesh.ftype)##weight
        gphi = self.pspaces[level].grad_basis(bc)
        cellmeasure = mesh.entity_measure('cell')

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        scaledArea = mu/rho*cellmeasure

        A11 = spdiags(np.repeat(scaledArea, 2), 0, 2*NC, 2*NC)

        phi = self.uspaces[level].basis(bc)
        A21 = np.einsum('ijm, km, i->ijk', gphi, phi, cellmeasure)

        cell2dof0 = self.uspaces[-1].cell_to_dof()
        ldof0 = self.uspaces[level].number_of_local_dofs()
        cell2dof1 = self.pspaces[level].cell_to_dof()
        ldof1 = self.pspaces[level].number_of_local_dofs()
		
        gdof0 = self.uspaces[level].number_of_global_dofs()
        gdof1 = self.pspaces[-1].number_of_global_dofs()
        I = np.einsum('ij, k->ijk', cell2dof1, np.ones(ldof0))
        J = np.einsum('ij, k->ikj', cell2dof0, np.ones(ldof1))

        A21 = csr_matrix((A21.flat, (I.flat, J.flat)), shape=(gdof1, gdof0))
        A12 = A21.transpose()

        A = bmat([(A11, A12), (A21, None)], format='csr', dtype=np.float)
        
        return A
        
    def get_right_vector(self, level):
        mesh = self.pspaces[level].mesh
        pde = self.pde
        mu = pde.mu
        rho = pde.rho
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        cellmeasure = mesh.entity_measure('cell')
        
        f = self.uspaces[level].source_vector(self.pde.f, self.integrator0, cellmeasure)
        b = self.pspaces[level].source_vector(self.pde.g, self.integrator1, cellmeasure)
        	
	## Neumann boundary condition
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        ec = mesh.entity_barycenter('edge')
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2node = mesh.ds.edge_to_node()
        bdEdge = edge[isBDEdge, :]
        d = np.sqrt(np.sum((node[edge2node[isBDEdge, 0], :]\
            - node[edge2node[isBDEdge, 1], :])**2, 1))
        mid = ec[isBDEdge, :]

        ii = np.tile(d*self.pde.neumann(mid)/2,(1,2))
        g = np.bincount(np.ravel(bdEdge,'F'), weights = np.ravel(ii), minlength=NN)
		
        g = g - b  

        b1 = np.r_[f, g]
        return b1
    

    def compute_initial_value(self):
        mesh = self.pspaces[-1].mesh
        pde = self.pde
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        cellmeasure = mesh.entity_measure('cell')
        
        up = np.zeros(2*NC+NN, dtype=np.float)
        idx = np.arange(2*NC+NN-1)
        up[idx] = spsolve(self.A[idx, :][:, idx], self.b[idx])

        u = up[:2*NC]
        p = up[2*NC:]
        c = np.sum(np.mean(p[cell], 1)*cellmeasure)/np.sum(cellmeasure)
        p -= c

        return u,p

    def prev_smoothing(self, u, p, level):
        pass

    def post_smoothing(self, u, p, level):
        pass

    def fas(self, ):
        u, p = self.compute_initial_value()
        while True:


