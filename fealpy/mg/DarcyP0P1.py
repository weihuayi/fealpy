import numpy as np
import csv
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
import scipy.sparse
from ..fem.integral_alg import IntegralAlg
from ..fem import doperator
from scipy.sparse.linalg import cg, inv, dsolve,spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace

class DarcyP0P1():
    def __init__(self, pde, mesh, p, integrator):
        self.femspace = LagrangeFiniteElementSpace(mesh, p)
        self.pde = pde
        self.mesh = self.femspace.mesh

#        self.uh = self.femspace.function()
#        self.uI = self.femspace.interpoletion(pde.solution)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = integrator
        self.integralalg = IntegralAlg(self.integrator, self.mesh, self.cellmeasure)

        self.node = mesh.node
        self.cell = mesh.ds.cell
        NC = mesh.number_of_cells()

    def gradbasis(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        node = self.node
        cell = self.cell

        ve1 = node[cell[:, 2],:] - node[cell[:, 1],:]
        ve2 = node[cell[:, 0],:] - node[cell[:, 2],:]
        ve3 = node[cell[:, 1],:] - node[cell[:, 0],:]
        area = 0.5*(-ve3[:, 0]*ve2[:, 1] + ve3[:, 1]*ve2[:, 0])

        Dlambda = np.zeros((NC,2,3))
        Dlambda[:,:,2] = np.c_[-ve3[:, 1]/(2*area), ve3[:, 0]/(2*area)]
        Dlambda[:,:,0] = np.c_[-ve1[:, 1]/(2*area), ve1[:, 0]/(2*area)]
        Dlambda[:,:,1] = np.c_[-ve2[:, 1]/(2*area), ve2[:, 0]/(2*area)]

        return Dlambda


    def get_left_matrix(self):
        femspace = self.femspace
        mesh = self.mesh
        cellmeasure = self.cellmeasure
        cell = mesh.ds.cell
        node = mesh.node

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        mu = self.pde.mu
        rho = self.pde.rho
        scaledArea = mu/rho*cellmeasure
        Dlambda = self.gradbasis()
        A11 = spdiags(np.r_[scaledArea,scaledArea], 0, 2*NC, 2*NC)

        ## Assemble gradient matrix for pressure
        I = np.arange(2*NC)
        data1 = Dlambda[:, 0, 0]*cellmeasure
        A12 = coo_matrix((data1, (I[:NC],cell[:, 0])), shape=(2*NC, NN))
        data2 = Dlambda[:, 1, 0]*cellmeasure
        A12 += coo_matrix((data2, (I[NC:],cell[:, 0])), shape=(2*NC, NN))
        
        data1 = Dlambda[:, 0, 1]*cellmeasure
        A12 += coo_matrix((data1, (I[:NC],cell[:, 1])), shape=(2*NC, NN))
        data2 = Dlambda[:, 1, 1]*cellmeasure
        A12 += coo_matrix((data2, (I[NC:],cell[:, 1])), shape=(2*NC, NN))

        data1 = Dlambda[:, 0, 2]*cellmeasure
        A12 += coo_matrix((data1, (I[:NC],cell[:, 2])), shape=(2*NC, NN))
        data2 = Dlambda[:, 1, 2]*cellmeasure
        A12 += coo_matrix((data2, (I[NC:],cell[:, 2])), shape=(2*NC, NN))
        A12 = A12.tocsr()
        A21 = A12.transpose()
        print('A12', A12.toarray())
        print('A21', A21.toarray())

        A = bmat([(A11, A12), (A21, None)], format='csr',dtype=np.float)

    
        return A

    def get_right_vector(self):
        mesh = self.mesh
        cellmeasure = self.cellmeasure
        node = mesh.node
        cell = mesh.ds.cell
        NN = mesh.number_of_nodes()

        bc = mesh.entity_barycenter('cell')
        ft = self.pde.f(bc)*np.c_[cellmeasure,cellmeasure]
        f = np.ravel(ft,'F')

        cell2edge = mesh.ds.cell_to_edge()
        ec = mesh.entity_barycenter('edge')
        mid1 = ec[cell2edge[:, 1],:]
        mid2 = ec[cell2edge[:, 2],:]
        mid3 = ec[cell2edge[:, 0],:]

        bt1 = cellmeasure*(self.pde.g(mid2) + self.pde.g(mid3))/6
        bt2 = cellmeasure*(self.pde.g(mid3) + self.pde.g(mid1))/6
        bt3 = cellmeasure*(self.pde.g(mid1) + self.pde.g(mid2))/6

        g = np.bincount(np.ravel(cell,'F'),weights=np.r_[bt1,bt2,bt3], minlength=NN)
        
        return np.r_[f,g]

    def solve(self):
        mesh = self.mesh
        node = self.node
        edge = mesh.ds.edge
        cell = mesh.ds.cell
        cellmeasure = self.cellmeasure
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        A = self.get_left_matrix()
        b = self.get_right_vector()

        ## Neumann boundary condition
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2node = mesh.ds.edge_to_node()
        bdEdge = edge[isBDEdge,:]
        ec = mesh.entity_barycenter('edge')
        d = np.sqrt(np.sum((node[edge2node[isBDEdge,0],:]\
                - node[edge2node[isBDEdge,1],:])**2,1))
        mid = ec[isBDEdge,:]

        ii = np.tile(d*self.pde.Neumann_boundary(mid)/2,(2,1))
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
