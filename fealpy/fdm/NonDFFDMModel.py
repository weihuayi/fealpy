import numpy as np
import time
import h5py
import pickle
import sys
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy import linalg as LA
from fealpy.fem.integral_alg import IntegralAlg
from scipy.sparse.linalg import cg, inv, dsolve, spsolve 

class NonDFFDMModel():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

        hx = mesh.hx
        hy = mesh.hy
        nx = mesh.ds.nx
        ny = mesh.ds.ny

        hx1 = hx.repeat(ny)
        hy1 = np.tile(hy,nx)
        hx2 = (hx[1:] + hx[:nx-1])/2#gain $hx_{i+1/2}$
        hy2 = (hy[1:] + hy[:ny-1])/2#gain $hy_{j+1/2}$
        hx3 = hx2.repeat(ny)
        hy3 = np.tile(hy2,nx)

        area0 = hx3*hy1[ny:] #area of all $hx_{i+1/2}*hy_{j}$
        area1 = hx1[nx:]*hy3 #area of all $hx_{i}*hy_{j+1/2}$
        area2 = hx1*hy1 #Area of all cell
        area1 = np.r_[area0,area1]

        self.hx1 = hx1
        self.hy1 = hy1
        self.hx3 = hx3
        self.hy3 = hy3
        self.area1 = area1
        self.area2 = area2

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

        nx = mesh.nx
        ny = mesh.ny

        hx1 = self.hx1
        hy1 = self.hy1
        hy3 = self.hy3
        hx3 = self.hx3


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

        C[flag] = 1/4/hx3*(hx1[L]*np.sqrt(uh0[flag]**2+uh0[P1]**2)\
                         + hx1[L]*np.sqrt(uh0[flag]**2+uh0[D1]**2)\
                         + hx1[R]*np.sqrt(uh0[flag]**2+uh0[P2]**2)\
                         + hx1[R]*np.sqrt(uh0[flag]**2+uh0[D2]**2))

        flag = ~isBDEdge & isXDEdge
        L = edge2cell[flag, 0]
        R = edge2cell[flag, 1]
        P1 = cell2edge[L, 3]
        D1 = cell2edge[L, 1]
        P2 = cell2edge[R, 3]
        D2 = cell2edge[R, 1]

        C[flag] = 1/4/hy3*(hy1[L]*np.sqrt(uh0[flag]**2+uh0[P1]**2)\
                         + hy1[L]*np.sqrt(uh0[flag]**2+uh0[D1]**2)\
                         + hy1[R]*np.sqrt(uh0[flag]**2+uh0[P2]**2)\
                         + hy1[R]*np.sqrt(uh0[flag]**2+uh0[D2]**2))
        
        C = mu/k + rho*beta*C

        return C

    def get_left_matrix(self):

        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        hx1 = self.hx1
        hy1 = self.hy1
        hx3 = self.hx3
        hy3 = self.hy3

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
        data = np.ones(len(I), dtype=ftype)/hx3

        A12 = coo_matrix((data, (I, R)), shape=(NE, NC))
        A12 += coo_matrix((-data, (I, L)), shape=(NE, NC))

        I, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        data = np.ones(len(I), dtype=ftype)/hy3
        A12 += coo_matrix((-data, (I, R)), shape=(NE, NC))
        A12 += coo_matrix((data, (I, L)), shape=(NE, NC))
        A12 = A12.tocsr()


        cell2edge = mesh.ds.cell_to_edge()
        I = np.arange(NC, dtype=itype)
        data = np.ones(NC, dtype=ftype)
#        A21 = coo_matrix((data/hx1, (I, cell2edge[:, 1])), shape=(NC, NE), dtype=ftype)
#        A21 += coo_matrix((-data/hx1, (I, cell2edge[:, 3])), shape=(NC, NE), dtype=ftype)
#        A21 += coo_matrix((data/hy1, (I, cell2edge[:, 2])), shape=(NC, NE), dtype=ftype)
#        A21 += coo_matrix((-data/hy1, (I, cell2edge[:, 0])), shape=(NC, NE), dtype=ftype)
        A21 = coo_matrix((hy1*data, (I, cell2edge[:, 1])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-hy1*data, (I, cell2edge[:, 3])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((hx1*data, (I, cell2edge[:, 2])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-hx1*data, (I, cell2edge[:, 0])), shape=(NC, NE), dtype=ftype)
        A21 = A21.tocsr()
        A = bmat([(A11, A12), (A21, None)], format='csr', dtype=ftype)

        return A

    def get_right_vector(self):
        pde = self.pde
        mesh = self.mesh
        hx1 = self.hx1
        hy1 = self.hy1

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
        b0[idx] = C[idx]*val #modify

        idx, = np.nonzero(isXDEdge & isBDEdge)
        val = pde.velocity_y(bc[idx])
        b0[idx] = C[idx]*val

        b1 =hx1*hy1*pde.source1(pc)
        return np.r_[b0, b1]

    


    def solve(self):
        mesh = self.mesh
        hx1 = self.hx1
        hy1 = self.hy1
        hx3 = self.hx3
        hy3 = self.hy3

        area1 = self.area1
        area2 = self.area2


        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        itype = mesh.itype
        ftype = mesh.ftype

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        I, = np.nonzero(~isBDEdge & isYDEdge)
        J, = np.nonzero(~isBDEdge & isXDEdge)
        idx = np.r_[I,J]

        start = time.time()
        b = self.get_right_vector()
        A = self.get_left_matrix()
        end = time.time()
        print('Assemble matrix time',end-start)

        tol = self.pde.tol
        ru = 1
        rp = 1
        eu = 1
        ep = 1
        count = 0
        iterMax = 2000
        r = np.zeros((2,iterMax),dtype=ftype)
        er = np.zeros((2,iterMax),dtype=ftype)
        

        while eu+ep > tol and count < iterMax:

            bnew = np.copy(b)
            
            x = np.r_[self.uh, self.ph]#The combination of self.uh and self.ph together
#            print('x',x)
            bnew = bnew - A@x

            # Modify matrix
            bdIdx = np.zeros((A.shape[0],), dtype = itype)
            bdIdx[NE] = 1

            Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
            T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
            AD = T@A@T + Tbd
 
            bnew[NE] = self.ph[0]
 
            x[:] = spsolve(AD, bnew)
#            print('b - Ax', LA.norm(bnew - AD@x)/LA.norm(bnew))
            u1 = x[:NE]
            p1 = x[NE:]

            eu = np.sqrt(np.sum(area1*(u1[idx]-self.uh0[idx])**2))
            ep = np.sqrt(np.sum(area2*(p1-self.ph0)**2))
#            print('eu',eu)
#            print('ep:',ep)

            self.uh0[:] = u1
            self.ph0[:] = p1
            b = self.get_right_vector()
            A = self.get_left_matrix()


            f = b[:NE]
            g = b[NE:]
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
            er[0,count] = eu
            er[0,count] = ep

            count = count + 1
#            print('ru:',ru)
#            print('rp:',rp)

        self.uh[:] = u1
        self.ph[:] = p1

        return count,r,self.uh,self.ph


    def get_max_error(self):
        ue = np.max(np.abs(self.uh - self.uI))
        pe = np.max(np.abs(self.ph - self.pI))
        return ue, pe

    def get_L2_error(self):
        mesh = self.mesh
        area1 = self.area1
        area2 = self.area2

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        I, = np.nonzero(~isBDEdge & isYDEdge)
        J, = np.nonzero(~isBDEdge & isXDEdge)
        idx = np.r_[I,J]

        ueL2 = np.sqrt(np.sum(area1*(self.uh[idx] - self.uI[idx])**2))
        peL2 = np.sqrt(np.sum(area2*(self.ph - self.pI)**2))
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
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        area1 = self.area1
        
        ftype = mesh.ftype
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        cell = np.arange(NC)
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
        DpeL2 = np.sqrt(np.sum(area1*(Dph[idx] - DpI[idx])**2))
        return DpeL2

    def get_Dp1L2_error(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        hx3 = self.hx3
        hy3 = self.hy3
        area1 = self.area1
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
        Dph[I] = (self.ph[R] - self.ph[L])/hx3

        J, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[J, 0]
        R = edge2cell[J, 1]
        DpI[J] = self.pde.grad_pressure_y(bc[J])
        Dph[J] = (self.ph[L] - self.ph[R])/hy3

        idx = np.r_[I,J]

        Dp1eL2 = np.sqrt(np.sum(area1*(Dph[idx] - DpI[idx])**2))

        return Dp1eL2

    def get_uqunorm_error(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        area1 = self.area1
        ftype = mesh.ftype

        mu = self.pde.mu
        k = self.pde.k
        rho = self.pde.rho
        beta = self.pde.beta

        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2cell = mesh.ds.edge_to_cell()
        normu = np.zeros(NE, dtype=ftype)
        C = self.get_nonlinear_coef()

        I, = np.nonzero(~isBDEdge & isYDEdge)
        normu[I] = self.pde.normu(bc[I])

        J, = np.nonzero(~isBDEdge & isXDEdge)
        normu[J] = self.pde.normu(bc[J])

        Qu = (C - mu/k)/rho/beta

        idx = np.r_[I,J]

        uqunorm = np.sqrt(np.sum(area1*((normu[idx] - Qu[idx])*self.uI[idx])**2))

        return uqunorm

    def get_uuqunorm_error(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        area1 = self.area1
        ftype = mesh.ftype

        mu = self.pde.mu
        k = self.pde.k
        rho = self.pde.rho
        beta = self.pde.beta

        bc = mesh.entity_barycenter('edge')
        normu = self.pde.normu(bc)

        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        isBDEdge = mesh.ds.boundary_edge_flag()
        C = self.get_nonlinear_coef()

        I, = np.nonzero(~isBDEdge & isYDEdge)
        J, = np.nonzero(~isBDEdge & isXDEdge)
        Qu = (C - mu/k)/rho/beta

        idx = np.r_[I,J]

        uuqunorm = np.sqrt(np.sum(area1*(normu[idx]*self.uI[idx]\
                   - Qu[idx]*self.uh[idx])**2))

        return uuqunorm

    def get_uqnorm_error(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        area1 = self.area1
        ftype = mesh.ftype

        mu = self.pde.mu
        k = self.pde.k
        rho = self.pde.rho
        beta = self.pde.beta

        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2cell = mesh.ds.edge_to_cell()
        normu = np.zeros(NE, dtype=ftype)
        C = self.get_nonlinear_coef()

        I, = np.nonzero(~isBDEdge & isYDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        normu[I] = self.pde.normu(bc[I])

        J, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[J, 0]
        R = edge2cell[J, 1]
        normu[J] = self.pde.normu(bc[J])

        idx = np.r_[I,J]

        Qu = (C - mu/k)/rho/beta

        uqnorm = np.sqrt(np.sum(area1*((normu[idx] - Qu[idx]))**2))

        return uqnorm
