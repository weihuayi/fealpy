import numpy as np
import time
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy.linalg import norm
from ..fem.integral_alg import IntegralAlg
from scipy.sparse.linalg import cg, inv, dsolve, spsolve 

class Forchheimer():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

        hx = mesh.hx
        hy = mesh.hy
        nx = mesh.ds.nx
        ny = mesh.ds.ny

        self.hx1 = hx.repeat(ny)
        self.hy1 = np.tile(hy,nx)
        hx2 = (hx[1:] + hx[:nx-1])# get $hx_{i+1/2}$
        hy2 = (hy[1:] + hy[:ny-1])# get $hy_{j+1/2}$
        self.hx3 = hx2.repeat(ny)
        self.hy3 = np.tile(hy2,nx)

        area00 = self.hx3*self.hy1[ny:] # area of all $hx_{i+1/2}*hy_{j}$
        area01 = self.hx1[nx:]*self.hy3 # area of all $hx_{i}*hy_{j+1/2}$
        self.area2 = self.hx1*self.hy1 # Area of all cells
        self.area1 = np.r_[area00, area01]

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        
        self.uh = np.zeros(NE, dtype=mesh.ftype) # the number solution of u
        self.ph = np.zeros(NC, dtype=mesh.ftype) # the number solution of p
        self.uh0 = np.zeros(NE, dtype=mesh.ftype) # Intermediate variables about u
        self.ph0 = np.zeros(NC, dtype=mesh.ftype) # Intermediate variables about p
        self.ph[0] = 1

    def get_nonlinear_coef(self):
        mesh = self.mesh
        uh0 = self.uh0

        nx = mesh.ds.nx
        ny = mesh.ds.ny

        hx1 = self.hx1
        hy1 = self.hy1
        hx3 = self.hx3
        hy3 = self.hy3

        itype = mesh.itype
        ftype = mesh.ftype

        mu = self.pde.mu
        k = self.pde.k
        rho = self.pde.rho
        beta = self.pde.beta

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        NE = mesh.number_of_edges()

        C = np.zeros(NE, dtype=ftype)
        edge2cell = mesh.ds.edge_to_cell()
        cell2edge = mesh.ds.cell_to_edge()

        flag = ~isBDEdge & isYDEdge
        L = edge2cell[flag, 0]
        R = edge2cell[flag, 1]
        P1 = cell2edge[L, 0]
        D1 = cell2edge[L, 2]
        P2 = cell2edge[R, 0]
        D2 = cell2edge[R, 2]

        C[flag] = 1/4/hx3*(hx1[L]*np.sqrt(uh0[flag]**2 + uh0[P1]**2)\
                         + hx1[L]*np.sqrt(uh0[flag]**2 + uh0[D1]**2)\
                         + hx1[R]*np.sqrt(uh0[flag]**2 + uh0[P2]**2)\
                         + hx1[R]*np.sqrt(uh0[flag]**2 + uh0[D2]**2))
                         
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

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        C = self.get_nonlinear_coef()
        A11 = spdiags(C,0,NE,NE)# correct
#        I, = np.nonzero(~isBDEdge & isYDEdge)
#        J, = np.nonzero(~isBDEdge & isXDEdge)
#        idx = np.r_[I,J]
#        A11 = coo_matrix((C[idx]*np.r_[hx3,hy3],(idx,idx)), shape=(NE,NE))
#
#        I, = np.nonzero(isBDEdge & isYDEdge)
#        J, = np.nonzero(isBDEdge & isXDEdge)
#        idx = np.r_[I,J]
#        A11 += coo_matrix((C[idx], (idx, idx)), shape=(NE, NE))

        edge2cell = mesh.ds.edge_to_cell()
        I, = np.nonzero(~isBDEdge & isYDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        data = np.ones(len(I), dtype=ftype)/hx3
#        data = np.ones(len(I), dtype=ftype)

        A12 = coo_matrix((data, (I, R)), shape=(NE, NC))
        A12 += coo_matrix((-data, (I, L)), shape=(NE, NC))

        I, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        data = np.ones(len(I), dtype=ftype)/hy3
#        data = np.ones(len(I), dtype=ftype)
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
#        A21 = A21.tocsr()
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
        hx3 = self.hx3
        hy3 = self.hy3

        itype = mesh.itype
        ftype = mesh.ftype

        NE = mesh.number_of_edges()
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

        b1 = hx1*hy1*pde.source1(pc)
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

        while eu+ep > tol and count < iterMax:

            bnew = np.copy(b)            
            x = np.r_[self.uh, self.ph]#The combination of self.uh and self.ph together
            bnew = bnew - A@x

            # Modify matrix
            bdIdx = np.zeros((A.shape[0],), dtype = itype)
            bdIdx[NE] = 1

            Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
            T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
            AD = T@A@T + Tbd
            bnew[NE] = self.ph[0]
       
            x[:] = spsolve(AD, bnew)
            u1 = x[:NE]
            p1 = x[NE:]
 #           p1[0] = p1[0] - np.mean(p1)

            eu = np.sqrt(np.sum(area1*(u1[idx]-self.uh0[idx])**2))
            ep = np.sqrt(np.sum(area2*(p1-self.ph0)**2))
           
            self.uh0[:] = u1
            self.ph0[:] = p1
            b = self.get_right_vector()
            A = self.get_left_matrix()
            f = b[:NE]
            g = b[NE:]
            A11 = A[:NE,:NE]
            A12 = A[:NE,NE:NE+NC]
            A21 = A[NE:NE+NC,:NE]
            if norm(f) == 0:
                ru = norm(f - A11@u1 - A12@p1)
            else:
                ru = norm(f - A11@u1 - A12@p1)/norm(f)

            if norm(g) == 0:
                rp = norm(g - A21@u1)
            else:
                rp = norm(g - A21@u1)/norm(g)


            r[0,count] = rp
            r[1,count] = ru

            count = count + 1

        self.uh[:] = u1
        self.ph[:] = p1
        print('uh',self.uh.shape)
        print('ph',self.ph.shape)
        print('ru:',ru)
        print('rp:',rp)
        print('eu:',eu)
        print('ep:',ep)
        print('solve matrix p and u')
        return count,self.uh


