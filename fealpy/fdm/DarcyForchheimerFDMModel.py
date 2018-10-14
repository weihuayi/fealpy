import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy import linalg as LA
from fealpy.fem.integral_alg import IntegralAlg
from fealpy.fdm.DarcyFDMModel import DarcyFDMModel
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
        umax = np.max(self.uI)
        pmax = np.max(np.abs(self.pI))
#        print('uI:',self.uI)
#        print('pI:',self.pI)
        pc = mesh.entity_barycenter('cell')
        self.pI = pde.pressure(pc)

        self.ph[0] = self.pI[0]
        pass

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
        A11 = spdiags(C,0,NE,NE).toarray()# correct


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
#        print('maxf',np.max(b0))
        return np.r_[b0, b1] 

    def solve(self):
        mesh = self.mesh

        hx = mesh.hx
        hy = mesh.hy
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        itype = mesh.itype
        ftype = mesh.ftype

        b = self.get_right_vector()
        A = self.get_left_matrix()

        tol = 1e-4
        ru = 1
        rp = 1
        count = 0
        iterMax = 2000
        while ru+rp > tol and count < iterMax:

            bnew = b
            
            x = np.r_[self.uh, self.ph]#把self.uh,self.ph组合在一起
            bnew = bnew - A@x

            # Modify matrix
            bdIdx = np.zeros((A.shape[0],), dtype = itype)
            bdIdx[NE] = 1

            Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
            T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
            AD = T@A@T + Tbd

            bnew[NE] = self.ph[0]

            # solve
            x[:] = spsolve(AD, bnew)
            u1 = x[:NE]
            p1 = x[NE:]

            f = b[:NE]
            g = b[NE:]

            ue = np.sqrt(np.sum(hx*hy*(u1-self.uh0)**2))
            pe = np.sqrt(np.sum(hx*hy*(p1-self.ph0)**2))
#            print('uh -u0:',ue)
#            print('ph - p0:',pe)

            self.uh0[:] = u1
            A = self.get_left_matrix()
            A11 = A[:NE,:NE]
            A12 = A[:NE,NE:NE+NC]
            A21 = A[NE:NE+NC,:NE]
            if LA.norm(f) == 0:
                ru = LA.norm(f - A11*u1 - A12*p1)
            else:
                ru = LA.norm(f - A11*u1 - A12*p1)/LA.norm(f)
            if LA.norm(g) == 0:
                rp = LA.norm(g - A21*u1)
            else:
                rp = LA.norm(g - A21*u1)/LA.norm(g)

            count = count + 1
 #           print('ru:',ru)
 #           print('rp:',rp)

        self.uh[:] = u1
        self.ph[:] = p1
#        print('uh:',self.uh)
        return count

    def grad_pressure(self):
        mesh = self.mesh
        ftype = mesh.ftype

        NC = mesh.number_of_cells()

        ph = self.ph
        hx = mesh.hx
        hy = mesh.hy

        val = np.zeros((NC,2),dtype=ftype)


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

    def get_H1_error(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        hx = mesh.hx
        hy = mesh.hy
        ueL2,peL2 = self.get_L2_error()
        ep = self.ph - self.pI
        psemi = np.sqrt(np.sum((ep[1:]-ep[:NC-1])**2))
        peH1 = peL2+psemi
        return ep,psemi

    def get_DpL2_error(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        hx = mesh.hx
        hy = mesh.hy
        Nx = int(1/hx)
        Ny = int(1/hy)
        ftype = mesh.ftype

        Dph = np.zeros((NC,2),dtype=ftype)
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')
        DpI = self.pde.grad_pressure(pc)

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        I, = np.nonzero(isBDEdge & isYDEdge)
        Dph[NC-Ny:NC,0] = self.pde.source2(bc[I[Ny:],:])
        J, = np.nonzero(isBDEdge & isXDEdge)
        Dph[Ny-1:NC:Ny,1] = self.pde.source3(bc[J[1::2],:])

        Dph[:NC-Ny,0] = (self.ph[Ny:] - self.ph[:NC-Ny])/hx
        
        m = np.arange(NC)
        m = m.reshape(Ny,Nx)
        n1 = m[:,1:].flatten()
        n2 = m[:,:Ny-1].flatten()
        Dph[n2,1] = (self.ph[n1] - self.ph[n2])/hy

        DpeL2 = np.sqrt(np.sum(hx*hy*(Dph[:] - DpI[:])**2))

        return DpeL2

