import numpy as np
import time
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy import linalg as LA
from ..fem.integral_alg import IntegralAlg
from scipy.sparse.linalg import cg, inv, dsolve, spsolve 

class DarcyForchheimerFDMModel():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        self.uh = np.zeros(NE, dtype=mesh.ftype)
        self.ph = np.zeros(NC, dtype=mesh.ftype)
        self.uI = np.zeros(NE, dtype=mesh.ftype) 
        self.uh0 = np.zeros(NE, dtype=mesh.ftype)
        self.ph0 = np.zeros(NC, dtype=mesh.ftype)
        
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        self.uI[isYDEdge] = pde.velocity_x(bc[isYDEdge])
        self.uI[isXDEdge] = pde.velocity_y(bc[isXDEdge]) 
        self.pI = pde.pressure(pc)

        self.ph[0] = self.pI[0]

    def get_nonlinear_coef(self):
        mesh = self.mesh
        uh0 = self.uh0

        itype = mesh.itype
        ftype = mesh.ftype

        mu = self.pde.mu
        k = self.pde.k

        rho = self.pde.rho
        beta = self.pde.beta

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        C = np.ones(NE, dtype=mesh.ftype)
        edge2cell = mesh.ds.edge_to_cell()
        cell2edge = mesh.ds.cell_to_edge()

        flag = ~isBDEdge & isYDEdge
        L = edge2cell[flag, 0]
        R = edge2cell[flag, 1]
        P1 = cell2edge[L, 0]# the 0 edge of the left cell
        D1 = cell2edge[L, 2]# the 2 edge of the left cell
        P2 = cell2edge[R, 0]# the 0 edge of the right cell
        D2 = cell2edge[R, 2]# the 2 edge of the right cell

        C[flag] = 1/4*(np.sqrt(uh0[flag]**2+uh0[P1]**2)+np.sqrt(uh0[flag]**2+uh0[D1]**2)\
                +np.sqrt(uh0[flag]**2+uh0[P2]**2)+np.sqrt(uh0[flag]**2+uh0[D2]**2))

        flag = ~isBDEdge & isXDEdge
        L = edge2cell[flag, 0]
        R = edge2cell[flag, 1]
        P1 = cell2edge[L, 3]
        D1 = cell2edge[L, 1]
        P2 = cell2edge[R, 3]
        D2 = cell2edge[R, 1]
        C[flag] = 1/4*(np.sqrt(uh0[flag]**2+uh0[P1]**2)+np.sqrt(uh0[flag]**2+uh0[D1]**2)\
                +np.sqrt(uh0[flag]**2+uh0[P2]**2)+np.sqrt(uh0[flag]**2+uh0[D2]**2))
        
        C = mu/k + rho*beta*C

        return C # no problem


    def get_left_matrix(self):

        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        itype = mesh.itype
        ftype = mesh.ftype

        idx = np.arange(NE)
        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        C = self.get_nonlinear_coef()
        A11 = spdiags(C,0,NE,NE)# correct

        edge2cell = mesh.ds.edge_to_cell()
        I, = np.nonzero(~isBDEdge & isYDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        data = np.ones(len(I), dtype=ftype)/mesh.hx

        A12 = coo_matrix((data, (I, R)), shape=(NE, NC))
        A12 += coo_matrix((-data, (I, L)), shape=(NE, NC))

        I, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        data = np.ones(len(I), dtype=ftype)/mesh.hy
        A12 += coo_matrix((-data, (I, R)), shape=(NE, NC))
        A12 += coo_matrix((data, (I, L)), shape=(NE, NC))
        A12 = A12.tocsr()

        
        cell2edge = mesh.ds.cell_to_edge()
        I = np.arange(NC, dtype=itype)
        data = np.ones(NC, dtype=ftype)
        A21 = coo_matrix((data/mesh.hx, (I, cell2edge[:, 1])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-data/mesh.hx, (I, cell2edge[:, 3])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((data/mesh.hy, (I, cell2edge[:, 2])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-data/mesh.hy, (I, cell2edge[:, 0])), shape=(NC, NE), dtype=ftype)
        A21 = A21.tocsr()
        A = bmat([(A11, A12), (A21, None)], format='csr', dtype=ftype)
        return A


    def get_right_vector(self):
        pde = self.pde
        mesh = self.mesh
        node = mesh.node
        itype = mesh.itype
        ftype = mesh.ftype

        NN = mesh.number_of_nodes()  
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        edge = mesh.entity('edge')
        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        mu = self.pde.mu
        k = self.pde.k
        rho = self.pde.rho
        beta = self.pde.beta

        ## Modify
        C = self.get_nonlinear_coef()#add

        b0 = np.zeros(NE, dtype=ftype)
        flag = ~isBDEdge & isYDEdge
        b0[flag] = pde.source2(bc[flag])
        flag = ~isBDEdge & isXDEdge
        b0[flag] = pde.source3(bc[flag])

        idx, = np.nonzero(isYDEdge & isBDEdge)
        val = pde.velocity_x(bc[idx])
        b0[idx] = (mu/k+beta*rho*C[idx])*val #modify

        idx, = np.nonzero(isXDEdge & isBDEdge)
        val = pde.velocity_y(bc[idx])
        b0[idx] = (mu/k+beta*rho*C[idx])*val

        b1 = pde.source1(pc)
        return np.r_[b0, b1] 

    def solve(self):
        mesh = self.mesh

        hx = mesh.hx
        hy = mesh.hy
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        itype = mesh.itype
        ftype = mesh.ftype

        start = time.time()
        b = self.get_right_vector()
        A = self.get_left_matrix()
        end = time.time()
        print('Construct linear system time:',end - start)
#        f = b[:NE]
#        g = b[NE:]
#        if LA.norm(f) == 0:
#            ru = LA.norm(f - A11*self.uh - A12*self.ph)
#        else:
#            ru = LA.norm(f - A11*self.uh - A12*self.ph)/LA.norm(f)
#        if LA.norm(g) == 0:
#            rp = LA.norm(g - A21*self.uh)
#        else:
#            rp = LA.norm(g - A21*self.uh)/LA.norm(g)

        tol = self.pde.tol
        ru = 1
        rp = 1
        eu = 1
        ep = 1
        count = 0
        iterMax = 2000
        r = np.zeros((2,iterMax),dtype=ftype)

        while eu+ep > tol and count < iterMax:

            bnew = b
            
            x = np.r_[self.uh, self.ph]#把self.uh,self.ph组合在一起
            bnew = bnew - A@x

            # Modify matrix
            bdIdx = np.zeros((A.shape[0],), dtype = itype)
            bdIdx[NE] = 1

            Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
            T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
            AD = T@A@T + Tbd

            idx1 = 1 - bdIdx
            idx2, = np.nonzero(idx1)
            x[idx2] = spsolve(AD[idx2,:][:,idx2], bnew[idx2])
            u1 = x[:NE]
            p1 = x[NE:]

            p1 = p1 - np.mean(p1)

            f = b[:NE]
            g = b[NE:]
            eu = np.sqrt(np.sum(hx*hy*(u1-self.uh0)**2))
            ep = np.sqrt(np.sum(hx*hy*(p1-self.ph0)**2))
            print('eu',eu)
            print('ep:',ep)

            self.uh0[:] = u1
            self.ph0[:] = p1
            A = self.get_left_matrix()
            A11 = A[:NE,:NE]
            A12 = A[:NE,NE:NE+NC]
            A21 = A[NE:NE+NC,:NE]
            if LA.norm(f) == 0:
                ru = LA.norm(f - A11@u1 - A12@p1)
            else:
                ru = LA.norm(f - A11@u1 - A12@p1)/LA.norm(f)
            if LA.norm(g) == 0:
                rp = LA.norm(g - A21@u1)
            else:
                rp = LA.norm(g - A21@u1)/LA.norm(g)


            r[0,count] = rp
            r[1,count] = ru
            b = self.get_right_vector()

            count = count + 1
            print('ru:',ru)
            print('rp:',rp)

        self.uh[:] = u1
        self.ph[:] = p1
        print('solve matrix p and u')
        return count,r


    def get_max_error(self):
        ue = np.max(np.abs(self.uh - self.uI))
        pe = np.max(np.abs(self.ph - self.pI))
        return ue, pe

    def get_L2_error(self):
        mesh = self.mesh
        hx = mesh.hx
        hy = mesh.hy
        ueL2 = np.sqrt(np.sum(hx*hy*(self.uh - self.uI)**2))
        peL2 = np.sqrt(np.sum(hx*hy*(self.ph - self.pI)**2))
        return ueL2,peL2

    def get_normu_error(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        hx = mesh.hx
        hy = mesh.hy

        mu = self.pde.mu
        k = self.pde.k

        rho = self.pde.rho
        beta = self.pde.beta

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        
        bc = mesh.entity_barycenter('edge')
        normu = self.pde.normu(bc)

        I, = np.nonzero(~isBDEdge & isYDEdge)
        J, = np.nonzero(~isBDEdge & isXDEdge)

        idx = np.r_[I,J]
        C = self.get_nonlinear_coef()
        normuh = (C - mu/k)/rho/beta

        normuL2 = np.sqrt(np.sum(hx*hy*((normu[idx] - normuh[idx])*self.uI[idx])**2))

        return normuL2

    def get_DpL2_error(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        hx = mesh.hx
        hy = mesh.hy
        ny = mesh.ds.ny
        nx = mesh.ds.nx
        ftype = mesh.ftype
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        DpI = np.zeros(NE, dtype=mesh.ftype)
        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        DpI[isYDEdge] = self.pde.grad_pressure_x(bc[isYDEdge])#modity
        DpI[isXDEdge] = self.pde.grad_pressure_y(bc[isXDEdge])

        cell2edge = mesh.ds.cell_to_edge()
        b = self.get_right_vector()
        C = self.get_nonlinear_coef()
        Dph = b[:NE] - C*self.uh

        I, = np.nonzero(~isBDEdge & isYDEdge)
        J, = np.nonzero(~isBDEdge & isXDEdge)
        idx = np.r_[I,J]
        DpeL2 = np.sqrt(np.sum(hx*hy*(Dph[idx] - DpI[idx])**2))
        return DpeL2

    def get_Dp1L2_error(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        hx = mesh.hx
        hy = mesh.hy
        nx = mesh.ds.nx
        ny = mesh.ds.ny
        ftype = mesh.ftype

        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2cell = mesh.ds.edge_to_cell()
        Dph = np.zeros(NE, dtype=ftype)
        DpI = np.zeros(NE, dtype=mesh.ftype)

        I, = np.nonzero(~isBDEdge & isYDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        DpI[I] = self.pde.grad_pressure_x(bc[I])
        Dph[I] = (self.ph[R] - self.ph[L])/hx

        J, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[J, 0]
        R = edge2cell[J, 1]
        DpI[J] = self.pde.grad_pressure_y(bc[J])
        Dph[J] = (self.ph[L] - self.ph[R])/hy

        Dp1eL2 = np.sqrt(np.sum(hx*hy*(Dph[:] - DpI[:])**2))

        return Dp1eL2
