import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy import linalg as LA
from fealpy.fem.integral_alg import IntegralAlg
from fealpy.fdm.DarcyFDMModel import DarcyFDMModel
from scipy.sparse.linalg import cg, inv, dsolve, spsolve

class DarcyForchheimerFDMModel():
    def __init__(self, pde1, pde, mesh):
        self.pde = pde
        self.pde1 = pde1
        self.mesh = mesh
        fdm = DarcyFDMModel(pde1, mesh)
        uu = fdm.solve()
        self.uu = uu

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        self.uh = np.zeros(NE, dtype=mesh.ftype)
        self.ph = np.zeros(NC, dtype=mesh.ftype)
        self.uI = np.zeros(NE, dtype=mesh.ftype) 
        
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        bc = mesh.entity_barycenter('edge')

        self.uI[isYDEdge] = pde.velocity_x(bc[isYDEdge])
        self.uI[isXDEdge] = pde.velocity_y(bc[isXDEdge]) 
        pc = mesh.entity_barycenter('cell')
        self.pI = pde.pressure(pc)

        self.ph[0] = self.pI[0]
        pass

    def get_Qu(self):
        mesh = self.mesh
        uu = self.uu

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

        I, = np.nonzero(~isBDEdge & isYDEdge)
        edge2cell = mesh.ds.edge_to_cell()
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        cell2edge = mesh.ds.cell_to_edge()
        P1 = cell2edge[L, 0]# the 0 edge of the left cell
        D1 = cell2edge[L, 2]# the 2 edge of the left cell
        P2 = cell2edge[R, 0]# the 0 edge of the right cell
        D2 = cell2edge[R, 2]# the 2 edge of the right cell

        Qu = np.zeros(NE//2,dtype=ftype)
        Qu[I] = 1/4*(np.sqrt(uu[I]**2+uu[P1]**2)+np.sqrt(uu[I]**2+uu[D1]**2)\
                +np.sqrt(uu[I]**2+uu[P2]**2)+np.sqrt(uu[I]**2+uu[D2]**2))
        idx = np.nonzero(isBDEdge & isYDEdge)
        Qu[idx] = 1
                
        Qu = mu/k + rho*beta*Qu

        return Qu
    def get_Qv(self):
        mesh = self.mesh
        uu = self.uu

        itype = mesh.itype
        ftype = mesh.ftype

        mu = self.pde.mu
        k = self.pde.k
        rho = self.pde.rho
        beta = self.pde.beta

        pde = self.pde
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        isBDEdge = mesh.ds.boundary_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        I, = np.nonzero(~isBDEdge & isXDEdge)
        edge2cell = mesh.ds.edge_to_cell()
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        cell2edge = mesh.ds.cell_to_edge()
        P1 = cell2edge[L, 3]
        D1 = cell2edge[L, 1]
        P2 = cell2edge[R, 3]
        D2 = cell2edge[R, 1]

        Qv = np.zeros(NE,dtype=ftype)

        Qv[I] = 1/4*(np.sqrt(uu[P1]**2+uu[I]**2)+np.sqrt(uu[D1]**2+uu[I]**2)+np.sqrt(uu[P2]**2+uu[I]**2)+np.sqrt(uu[D2]**2+uu[I]**2))
        idx = np.nonzero(isBDEdge & isXDEdge)
        Qv[idx] = 1

        Qv = mu/k + rho*beta*Qv
        
        return Qv

    def get_Qu1(self):
        mesh = self.mesh

        itype = mesh.itype
        ftype = mesh.ftype
        NE = mesh.number_of_edges()

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        I, = np.nonzero(~isBDEdge & isYDEdge)

        edge2cell = mesh.ds.edge_to_cell()
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]

        cell2edge = mesh.ds.cell_to_edge()
        P1 = cell2edge[L, 0]# the 0 edge of the left cell
        D1 = cell2edge[L, 2]# the 2 edge of the left cell
        P2 = cell2edge[R, 0]# the 0 edge of the right cell
        D2 = cell2edge[R, 2]# the 2 edge of the right cell
        
        Qu1 = np.zeros(NE//2,dtype = ftype)
        bc = mesh.entity_barycenter('edge')
        pi = np.pi
        Qu1[I] = 1/4*(np.sqrt((np.sin(pi*bc[I,0])*np.cos(pi*bc[I,1]))**2+(np.cos(pi*bc[P1,0])*np.sin(pi*bc[P1,1]))**2) \
                +np.sqrt((np.sin(pi*bc[I,0])*np.cos(pi*bc[I,1]))**2+(np.cos(pi*bc[D1,0])*np.sin(pi*bc[D1,1]))**2) \
            +np.sqrt((np.sin(pi*bc[I,0])*np.cos(pi*bc[I,1]))**2+(np.cos(pi*bc[P2,0])*np.sin(pi*bc[P2,1]))**2)\
            +np.sqrt((np.sin(pi*bc[I,0])*np.cos(pi*bc[I,1]))**2+(np.cos(pi*bc[D2,0])*np.sin(pi*bc[D2,1]))**2))
        
        return Qu1

    def get_Qv1(self):

        mesh = self.mesh

        itype = mesh.itype
        ftype = mesh.ftype
        NE = mesh.number_of_edges()

        isBDEdge = mesh.ds.boundary_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        I, = np.nonzero(~isBDEdge & isXDEdge)

        edge2cell = mesh.ds.edge_to_cell()
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]

        cell2edge = mesh.ds.cell_to_edge()
        P1 = cell2edge[L, 3]# the 3 edge of the left cell
        D1 = cell2edge[L, 1]# the 1 edge of the left cell
        P2 = cell2edge[R, 3]# the 3 edge of the right cell
        D2 = cell2edge[R, 1]# the 1 edge of the right cell
        
        Qv1 = np.zeros(NE, dtype = ftype)
        bc = mesh.entity_barycenter('edge')
        pi = np.pi
        Qv1[I] =1/4*(np.sqrt((np.sin(pi*bc[P1,0])*np.cos(pi*bc[P1,1]))**2+(np.cos(pi*bc[I,0])*np.sin(pi*bc[I,1]))**2) \
                +np.sqrt((np.sin(pi*bc[D1,0])*np.cos(pi*bc[D1,1]))**2+(np.cos(pi*bc[I,0])*np.sin(pi*bc[I,1]))**2) \
            +np.sqrt((np.sin(pi*bc[P2,0])*np.cos(pi*bc[P2,1]))**2+(np.cos(pi*bc[I,0])*np.sin(pi*bc[I,1]))**2)\
            +np.sqrt((np.sin(pi*bc[D2,0])*np.cos(pi*bc[D2,1]))**2+(np.cos(pi*bc[I,0])*np.sin(pi*bc[I,1]))**2))
        Qv1 = Qv1[NE//2:NE]

        return Qv1


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

        Qu = self.get_Qu()
        Qv = self.get_Qv()
        Qv = Qv[NE//2:NE]
        Quv = np.hstack((Qu, Qv))
        A11 = np.diag(Quv)


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

        Qu = self.get_Qu()
        Qv = self.get_Qv()
        Qu1 = self.get_Qu1()
        Qv1 = self.get_Qv1()

        mu = self.pde.mu
        k = self.pde.k
        rho = self.pde.rho
        beta = self.pde.beta

        b = np.zeros(NE+NC, dtype=ftype)

        b[:NE//2] = pde.source2(bc[0:NE//2,:], Qu1)
        b[NE//2:NE] = pde.source3(bc[NE//2:,:], Qv1)


        idx, = np.nonzero(isYDEdge & isBDEdge)
        val = pde.velocity_x(bc[idx, :])
        
        b[idx] = (mu/k+rho*beta*Qu[idx])*val

        idx, = np.nonzero(isXDEdge & isBDEdge)
        val = pde.velocity_y(bc[idx, :])
        b[idx] = (mu/k+rho*beta*Qv[idx])*val


        b[NE:] = pde.source1(pc)
        
        return b

    def solve(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        itype = mesh.itype
        ftype = mesh.ftype

        A = self.get_left_matrix()
        b = self.get_right_vector()

        x = np.r_[self.uh, self.ph]#把self.uh,self.ph组合在一起
        b = b - A@x

        # Modify matrix
        bdIdx = np.zeros((A.shape[0],), dtype = itype)
        bdIdx[NE] = 1

        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
        AD = T@A@T + Tbd

        b[NE] = self.ph[0]

        # solve
        x[:] = spsolve(AD, b)
        return x

    def cycling(self):
        mesh = self.mesh
        hx = mesh.hx
        hy = mesh.hy

        NE = mesh.number_of_edges()
        uu = self.uu
        u2 = uu[:NE]
        p2 = uu[NE:]

        tol = 1e-6
        ru = 1
        rp = 1
        count = 0
        iterMax = 2000
        while ru+rp > tol and count < iterMax:
            u  = self.solve()
            u1 = u[:NE]
            p1 = u[NE:]

            A = self.get_left_matrix()
            b = self.get_right_vector()
            A11 = A[:NE,:NE]
            A12 = A[:NE,NE:]
            A21 = A[NE:,:NE]
            f = b[:NE]
            g = b[NE:]

            ru = np.sqrt(np.sum(hx*hy*(u1-u2)**2))
            rp = np.sqrt(np.sum(hx*hy*(p1-p2)**2))
#            if LA.norm(f) == 0:
#                ru = LA.norm(f - A11*u1 - A12*p1)
#            else:
#                ru = LA.norm(f - A11*u1 - A12*p1)/LA.norm(f)
#            if LA.norm(g) == 0:
#                rp = LA.norm(g - A21*u1)
#            else:
#                rp = LA.norm(g - A21*u1)/LA.norm(g)
#
            uu[:NE] = u1
            uu[NE:] = p1
            Qu = self.get_Qu()
            Qv = self.get_Qv()
            Qu1 = self.get_Qu1()
            Qv1 = self.get_Qv1()
            count = count + 1
            self.uh = u1
            self.ph = p1
            print('ru',ru)
            print('rp',rp)

            return count

    
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

 
