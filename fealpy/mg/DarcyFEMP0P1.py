import numpy as np
import csv
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
import scipy.sparse
from ..fem.integral_alg import IntegralAlg
from ..fem import doperator
from scipy.sparse.linalg import cg, inv, dsolve,spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace

class DarcyFEMP0P1():
    def __init__(self, pde, mesh, integrator0, integrator1):
        self.space0 = VectorLagrangeFiniteElementSpace(mesh, 0, spacetype='D')
        self.space1 = LagrangeFiniteElementSpace(mesh, 1, spacetype='C')
        self.pde = pde
        self.mesh = self.space0.mesh

        self.uh = self.space0.function()
        self.ph = self.space1.function()

#        self.uI = self.femspace.function()
#        self.pI = self.femspace.interpolation(pde.pressure)

        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator1 = integrator1
        self.integrator0 = integrator0

        self.integralalg1 = IntegralAlg(self.integrator1, self.mesh, self.cellmeasure)
        self.integralalg0 = IntegralAlg(self.integrator0, self.mesh, self.cellmeasure)


    def get_left_matrix(self):
        space1 = self.space1
        space0 = self.space0
        mesh = self.mesh
        cellmeasure = self.cellmeasure
        cell = mesh.ds.cell
        node = mesh.node

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        mu = self.pde.mu
        rho = self.pde.rho
	    
        bc = np.array([1/3, 1/3, 1/3], dtype=mesh.ftype)
        phi = space0.basis(bc)
        gphi = space1.grad_basis(bc)
	    
        scaledArea = mu/rho*cellmeasure

        A11 = spdiags(np.repeat(scaledArea, 2), 0, 2*NC, 2*NC)

        A21 = np.einsum('ijm, km, i->ijk', gphi, phi, cellmeasure)

        cell2dof0 = space0.cell_to_dof()
        ldof0 = space0.number_of_local_dofs()
        cell2dof1 = space1.cell_to_dof()
        ldof1 = space1.number_of_local_dofs()

        gdof0 = space0.number_of_global_dofs()
        gdof1 = space1.number_of_global_dofs()
        I = np.einsum('ij, k->ijk', cell2dof1, np.ones(ldof0))
        J = np.einsum('ij, k->ikj', cell2dof0, np.ones(ldof1))
	    
        A21 = csr_matrix((A21.flat, (I.flat, J.flat)), shape=(gdof1, gdof0))
        A12 = A21.transpose() 
        A = bmat([(A11, A12), (A21, None)], format='csr',dtype=np.float)
        return A


    def get_right_vector(self):
        mesh = self.mesh
        cellmeasure = self.cellmeasure
        node = mesh.node
        edge = mesh.ds.edge
        cell = mesh.ds.cell
        NN = mesh.number_of_nodes()

        bc = mesh.entity_barycenter('cell')
        ft = self.pde.f(bc)*np.c_[cellmeasure,cellmeasure]
        f = ft.flatten()

        cell2edge = mesh.ds.cell_to_edge()
        ec = mesh.entity_barycenter('edge')
        mid1 = ec[cell2edge[:, 1],:]
        mid2 = ec[cell2edge[:, 2],:]
        mid3 = ec[cell2edge[:, 0],:]

        ## need modify
        bt1 = cellmeasure*(self.pde.g(mid2) + self.pde.g(mid3))/6
        bt2 = cellmeasure*(self.pde.g(mid3) + self.pde.g(mid1))/6
        bt3 = cellmeasure*(self.pde.g(mid1) + self.pde.g(mid2))/6

        g = np.bincount(np.ravel(cell),weights=np.r_[bt1,bt2,bt3], minlength=NN)
        
        return np.r_[f,g]

    def solve(self):
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')
        cellmeasure = self.cellmeasure
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        A = self.get_left_matrix()
        A11 = A[:2*NC, :2*NC]
        A12 = A[:2*NC, 2*NC:]
        A21 = A[2*NC:, :2*NC]
        #A12 = A21.transpose()
        b = self.get_right_vector()

        ## Neumann boundary condition
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2node = mesh.ds.edge_to_node()
        bdEdge = edge[isBDEdge,:]
        ec = mesh.entity_barycenter('edge')
        d = np.sqrt(np.sum((node[edge2node[isBDEdge,0],:]\
                - node[edge2node[isBDEdge,1],:])**2,1))
        mid = ec[isBDEdge,:]

        ii = np.tile(d*self.pde.neumann(mid)/2,(2,1))
        g = np.bincount(np.ravel(bdEdge,'F'),\
                weights=np.ravel(ii), minlength=NN)
        g = g - b[2*NC:]

        ## Direct Solver
        b1 = np.r_[b[:2*NC],g]
        up = np.zeros(2*NC+NN,dtype=np.float)
        idx = np.arange(2*NC+NN-1)
        up[idx] = spsolve(A[idx,:][:,idx],b1[idx])
        u = up[:2*NC]
        p = up[2*NC:]
        c = np.sum(np.mean(p[cell],1)*cellmeasure)/np.sum(cellmeasure)
        p = p - c

        return u,p
