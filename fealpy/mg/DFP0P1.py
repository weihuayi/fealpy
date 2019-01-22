import numpy as np
import csv
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
import scipy.sparse
from ..fem.integral_alg import IntegralAlg
from numpy.linalg import norm
from ..fem import doperator
from ..mg.DarcyFEMP0P1 import DarcyFEMP0P1
from scipy.sparse.linalg import cg, inv, dsolve,spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from timeit import default_timer as timer

class DFP0P1():
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
        self.lfem = DarcyFEMP0P1(self.pde, self.mesh, integrator0, integrator1)
        self.uh0,self.ph0 = self.lfem.solve()
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
       
        return A11, A21

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
        f2 = self.space0.source_vector(self.pde.f, self.integrator0, cellmeasure)# f == f2

        cell2edge = mesh.ds.cell_to_edge()
        ec = mesh.entity_barycenter('edge')
        mid1 = ec[cell2edge[:, 1],:]
        mid2 = ec[cell2edge[:, 2],:]
        mid3 = ec[cell2edge[:, 0],:]

        ## need modify
        bt1 = cellmeasure*(self.pde.g(mid2) + self.pde.g(mid3))/6
        bt2 = cellmeasure*(self.pde.g(mid3) + self.pde.g(mid1))/6
        bt3 = cellmeasure*(self.pde.g(mid1) + self.pde.g(mid2))/6

        b = np.bincount(np.ravel(cell),weights=np.r_[bt1,bt2,bt3], minlength=NN)
        ##
        b2 = self.space1.source_vector(self.pde.g, self.integrator1, cellmeasure)
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2node = mesh.ds.edge_to_node()
        bdEdge = edge[isBDEdge,:]
        d = np.sqrt(np.sum((node[edge2node[isBDEdge,0],:]\
                - node[edge2node[isBDEdge, 1],:])**2,1))
        mid = ec[isBDEdge, :]
        data = d*self.pde.neumann(mid)/2
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
        A11, A21 = self.get_left_matrix()
        A12 = A21.transpose()
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
        area = np.repeat(cellmeasure, 2)
        ##  Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)

        F = self.uh0/alpha - (mu/rho)*self.uh0 - (A12@self.ph0 - b[:2*NC])/area
#        print('m1',(A12@self.ph0))        
#        print('F3',F)
        FL = np.sqrt(F[::2]**2 + F[1::2]**2)
        gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
        uhalf = F/np.repeat(gamma, 2)
        ## Direct Solver 

        Aalpha = A11 + spdiags(area/alpha, 0, 2*NC,2*NC)

        while eu+ep > tol and n < maxN:
            ## solve the linear Darcy equation
            #uhalfL = np.sqrt(uhalf[:NC]**2 + uhalf[NC:]**2)
            uhalfL = np.sqrt(uhalf[::2]**2 + uhalf[1::2]**2)
            fnew = b[:2*NC] + uhalf*area/alpha\
                    - beta/rho*uhalf*np.repeat(uhalfL, 2)*area

            ## Direct Solver
            Aalphainv = spdiags(1/Aalpha.data, 0, 2*NC, 2*NC)
            Ap = A21@Aalphainv@A12
           # print('Ap',Ap.toarray())
            bp = A21@(Aalphainv@fnew) - b[2*NC:]
           # print('bp', bp)
            p1 = np.zeros(NN,dtype=np.float)
            p1[1:] = spsolve(Ap[1:,1:],bp[1:])
            c = np.sum(np.mean(p1[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p1 = p1 - c
            u1 = Aalphainv@(fnew - A12@p1)

            ## Step1:Solve the nonlinear Darcy equation

            F = u1/alpha - (mu/rho)*u1 - (A12@p1 - b[:2*NC])/area
            FL = np.sqrt(F[::2]**2 + F[1::2]**2)
            gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
            uhalf = F/np.repeat(gamma, 2)

            ## Updated residual and error of consective iterations
            r[0,n] = ru
            r[1,n] = rp
            n = n + 1
            uLength = np.sqrt(u1[::2]**2 + u1[1::2]**2)
            Lu = A11@u1 + (beta/rho)*np.repeat(uLength*cellmeasure, 2)*u1 + A12@p1
            ru = norm(b[:2*NC] - Lu)/norm(b[:2*NC])
            if norm(b[2*NC:]) == 0:
                rp = norm(b[2*NC:] - A21@u1)
            else:
                rp = norm(b[2*NC:] - A21@u1)/norm(b[2*NC:])
            eu = np.max(abs(u1 - self.uh0))
            ep = np.max(abs(p1 - self.ph0))

            self.uh0[:] = u1
            self.ph0[:] = p1


        self.uh[:] = u1
        self.ph[:] = p1
        return u1, p1 


        

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
