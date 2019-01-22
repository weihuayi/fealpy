import numpy as np
import csv
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
import scipy.sparse
from ..fem.integral_alg import IntegralAlg
from numpy.linalg import norm
from ..fem import doperator
from ..mg.DarcyP0P1 import DarcyP0P1
from scipy.sparse.linalg import cg, inv, dsolve,spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from timeit import default_timer as timer

class DarcyForchheimerP0P1():
    def __init__(self, pde, mesh, integrator0, integrator1):
        self.space0 = VectorLagrangeFiniteElementSpace(mesh, 0, spacetype='D')
        self.space1 = LagrangeFiniteElementSpace(mesh, 1, spacetype='C')
        self.pde = pde
        self.mesh = mesh

        self.uh = self.space0.function()
        self.ph = self.space1.function()

        self.uh0 = self.space0.function()
        self.ph0 = self.space1.function()

#        self.uI = self.femspace.function()
#        self.pI = self.femspace.interpolation(pde.pressure)

        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator1 = integrator1
        self.integrator0 = integrator0
        self.lfem = DarcyP0P1(self.pde, self.mesh, 1, integrator1)
        self.uh0,self.ph0 = self.lfem.solve()
        self.integralalg1 = IntegralAlg(self.integrator1, self.mesh, self.cellmeasure)
        self.integralalg0 = IntegralAlg(self.integrator0, self.mesh, self.cellmeasure)
        
    def gradbasis(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        node = mesh.node
        cell = mesh.ds.cell

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
        space1 = self.space1
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
        f = np.ravel(ft,'F')

        cell2edge = mesh.ds.cell_to_edge()
        ec = mesh.entity_barycenter('edge')
        mid1 = ec[cell2edge[:, 1],:]
        mid2 = ec[cell2edge[:, 2],:]
        mid3 = ec[cell2edge[:, 0],:]

        bt1 = cellmeasure*(self.pde.g(mid2) + self.pde.g(mid3))/6
        bt2 = cellmeasure*(self.pde.g(mid3) + self.pde.g(mid1))/6
        bt3 = cellmeasure*(self.pde.g(mid1) + self.pde.g(mid2))/6

        b = np.bincount(np.ravel(cell,'F'),weights=np.r_[bt1,bt2,bt3], minlength=NN)

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
        g = g - b

        
        return np.r_[f,g]

    def solve(self):
        mesh = self.mesh
        node = mesh.node
        edge = mesh.ds.edge
        cell = mesh.ds.cell
        cellmeasure = self.cellmeasure
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        A = self.get_left_matrix()
        A11 = A[:2*NC,:2*NC]
        A12 = A[:2*NC,2*NC:]
        A21 = A[2*NC:,:2*NC]
        b = self.get_right_vector()

        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.pde.tol
        maxN = self.pde.maxN
        ru = 1
        rp = 1
        eu = 1
        ep = 1

        ## P-R iteration for D-F equation
        n = 0
        r = np.ones((2,maxN),dtype=np.float)
        area = np.r_[cellmeasure,cellmeasure]
        
        ##  Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)

        F = self.uh0/alpha - (mu/rho)*self.uh0 - (A12@self.ph0 - b[:2*NC])/area
#        print('ua2', self.uh0/alpha)
#        print('mu2', (mu/rho)*self.uh0)
#        print('m2', (A12@self.ph0))
#        print('Ab1', A12@self.ph0 - b[:2*NC])
#        print('F1',F[:NC])
#        print('F2',F[NC:])
        FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
        gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
        uhalf = F/np.r_[gamma,gamma]
        ## Direct Solver 

        Aalpha = A11 + spdiags(area/alpha, 0, 2*NC,2*NC)

        while eu+ep > tol and n < maxN:
            ## solve the linear Darcy equation
            uhalfL = np.sqrt(uhalf[:NC]**2 + uhalf[NC:]**2)
            fnew = b[:2*NC] + uhalf*area/alpha\
                    - beta/rho*uhalf*np.r_[uhalfL,uhalfL]*area
            ## Direct Solver
            Aalphainv = spdiags(1/Aalpha.data, 0, 2*NC, 2*NC)
            Ap = A21@Aalphainv@A12
            bp = A21@(Aalphainv@fnew) - b[2*NC:]

            p1 = np.zeros(NN,dtype=np.float)
            p1[1:] = spsolve(Ap[1:,1:],bp[1:])
            c = np.sum(np.mean(p1[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p1 = p1 - c
            u1 = Aalphainv@(fnew - A12@p1)

            ## Step1:Solve the nonlinear Darcy equation

            F = u1/alpha - (mu/rho)*u1 - (A12@p1 - b[:2*NC])/area
            FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
            gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
            uhalf = F/np.r_[gamma,gamma]

            ## Updated residual and error of consective iterations
            r[0,n] = ru
            r[1,n] = rp
            n = n + 1
            uLength = np.sqrt(u1[:NC]**2 + u1[NC:]**2)
            Lu = A11@u1 + (beta/rho)*np.tile(uLength*cellmeasure,(1,2))*u1 + A12@p1
            ru = norm(b[:2*NC] - Lu)/norm(b[:2*NC])
            if norm(b[2*NC:]) == 0:
                rp = norm(b[2*NC:] - A21@u1)
            else:
                rp = norm(b[2*NC:] - A21@u1)/norm(b[2*NC:])
            eu = np.max(abs(u1 - self.uh0))
            ep = np.max(abs(p1 - self.ph0))

            self.uh0[:] = u1
            self.ph0[:] = p1

            
        u11 = u1[:NC]
        u22 = u1[NC:]
        u12 = np.c_[u11,u22].flatten()

        self.uh[:] = u12
        self.ph[:] = p1
        return u12, p1


    def get_residual_estimate(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2cell = mesh.ds.edge_to_cell()
        uh = self.uh
        ph = self.ph
        pde = self.pde

        bc = np.array([1/3, 1/3, 1/3], dtype=mesh.ftype)

        u0 = uh.value(bc)
        gp = ph.grad_value(bc)

        lu = np.sqrt(np.sum(u0**2, axis=1))
        J = (pde.mu/pde.rho + pde.beta/pde.rho*lu)*u0 + gp
        n, t = mesh.edge_frame()
        l = mesh.entity_measure('edge')

        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])

        J1 = np.zeros(NE, dtype=mesh.ftype)
        J2 = np.zeros(NE, dtype=mesh.ftype)
        J3 = np.zeros(NE, dtype=mesh.ftype)

        J1[isBdEdge] = np.sum(J[edge2cell[isBdEdge, 0]]*n[isBdEdge], axis=-1)
        j = J[edge2cell[~isBdEdge, 0]] - J[edge2cell[~isBdEdge, 1]] 
        J1[~isBdEdge] = np.sum(j*n[~isBdEdge], axis=-1)
        J2[~isBdEdge] = np.sum(j*t[~isBdEdge], axis=-1)
        J3[~isBdEdge] = np.sum((u0[edge2cell[~isBdEdge, 0]] - u0[edge2cell[~isBdEdge, 1]])*n[~isBdEdge], axis=-1)

        
        


        

    def get_uL2_error(self):
        
        uh = self.uh.value
        u = self.pde.velocity

        uL2 = self.integralalg0.L2_error(u,uh)
        return uL2

    def get_pL2_error(self):
        p = self.pde.pressure
        ph = self.ph.value
        pL2 = self.integralalg1.L2_error(p,ph)
        return pL2


    def get_H1_error(self):
        mesh = self.mesh
        gp = self.pde.grad_pressure
        gph = self.ph.grad_value
        H1 = self.integralalg1.L2_error(gp, gph)
        return H1
