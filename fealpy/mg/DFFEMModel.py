import numpy as np
import csv
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
import scipy.sparse
from fealpy.fem.integral_alg import IntegralAlg
from numpy.linalg import norm
from fealpy.fem import doperator
from fealpy.mg.DarcyFEMModel import DarcyP0P1
from scipy.sparse.linalg import cg, inv, dsolve,spsolve
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace

class DarcyForchheimerP0P1():
    def __init__(self, pde, mesh, integrator):
        self.space0 = VectorLagrangeFiniteElementSpace(mesh, 0, spacetype='D')
        self.space1 = LagrangeFiniteElementSpace(mesh, 1, spacetype='C') 
        self.pde = pde
        self.mesh = self.femspace.mesh

        self.uh = self.space0.function()
        self.ph = self.space1.function()

        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = integrator
        self.lfem = DarcyP0P1(self.pde, self.mesh, p, integrator)
        self.uh0,self.ph0 = self.lfem.solve()
        self.integralalg = IntegralAlg(self.integrator, self.mesh, self.cellmeasure)

    def get_left_matrix(self):


        bc = np.array([1/3, 1/3, 1/3], dtype=self.ftype)
        phi = self.space0.basis(bc)
        gphi = self.space1.grad_basis(bc)
        A21 = np.einsum('ijm, km, i->ijk', gphi, phi, self.cellmeasure)

        cell2dof0 = self.space0.cell_to_dof()
        ldof0 = self.space0.number_of_local_dofs()
        cell2dof1 = self.space1.cell_to_dof()
        ldof1 = space.number_of_local_dofs()
        I = np.einsum('ij, k->ijk', cell2dof1, np.ones(ldof0))
        J = np.einsum('ij, k->ikj', cell2dof0, np.ones(ldof1))
        gdof = space.number_of_global_dofs()
        # Construct the stiffness matrix
        A21 = csr_matrix((A12.flat, (I.flat, J.flat)), shape=(gdof, gdof))
         
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
        ii = np.tile(d*self.pde.Neumann_boundary(mid)/2,(1,2))

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

        ## P-R iteration for D-F equation
        n = 0
        r = np.ones((2,maxN),dtype=np.float)
        area = np.r_[cellmeasure,cellmeasure]
        ##  Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)

        F = self.uh0/alpha - (mu/rho)*self.uh0 - (A12@self.ph0 - b[:2*NC])/area
        FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
        gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
        uhalf = F/np.r_[gamma,gamma]
        ## Direct Solver 

        Aalpha = A11 + spdiags(area/alpha, 0, 2*NC,2*NC)

        while ru+rp > tol and n < maxN:
            ## solve the linear Darcy equation
            uhalfL = np.sqrt(uhalf[:NC]**2 + uhalf[NC:]**2)
            fnew = b[:2*NC] + uhalf*area/alpha\
                    - beta/rho*uhalf*np.r_[uhalfL,uhalfL]*area

            ## Direct Solver
            Aalphainv = inv(Aalpha)
            Ap = A21@Aalphainv@A12
            bp = A21@(Aalphainv@fnew) - b[2*NC:]
            p = np.zeros(NN,dtype=np.float)
            p[1:] = spsolve(Ap[1:,1:],bp[1:])
            c = np.sum(np.mean(p[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p = p - c
            u = Aalphainv@(fnew - A12@p)

            ## Step1:Solve the nonlinear Darcy equation

            F = u/alpha - (mu/rho)*u - (A12@p - b[:2*NC])/area
            FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
            gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
            uhalf = F/np.r_[gamma,gamma]

            ## Updated residual and error of consective iterations

            n = n + 1
            uLength = np.sqrt(u[:NC]**2 + u[NC:]**2)
            Lu = A11@u + (beta/rho)*np.tile(uLength*cellmeasure,(1,2))*u + A12@p
            ru = norm(b[:2*NC] - Lu)/norm(b[:2*NC])
            if norm(b[2*NC:]) == 0:
                rp = norm(b[2*NC:] - A21@u)
            else:
                rp = norm(b[2*NC:] - A21@u)/norm(b[2*NC:])

            self.uh0 = u
            self.ph0 = p
            r[0,n] = ru
            r[1,n] = rp

        self.u = u
        self.p = p
        return u,p

    def get_pL2_error(self):

        p = self.pde.pressure
        ph = self.ph.value

        pL2 = self.integralalg.L2_error(p,ph)
        return pL2

    def get_uL2_error(self):
        mesh = self.mesh
        bc = mesh.entity_barycenter('cell')
        uI = self.pde.velocity(bc)
        
        uh = self.uh.value
        u = self.pde.velocity



#        self.integralalg = IntegralAlg(self.integrator, self.mesh, self.cellmeasure)
        uL2 = self.integralalg.L2_error(u,uh)
        return uL2


    def get_H1_error(self):
        mesh = self.mesh
        gp = self.pde.grad_pressure
        u,p = self.solve()
        gph = p.grad_value
        H1 = self.integralalg.L2_error(gp, gph)
        return H1
