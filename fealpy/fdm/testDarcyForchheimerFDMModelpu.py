import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy import linalg as LA
from fealpy.fem.integral_alg import IntegralAlg
from scipy.sparse.linalg import cg, inv, dsolve, spsolve

class DarcyForchheimerFDMModel():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        self.uh = np.zeros(NE, dtype=mesh.ftype)
        self.ph = np.zeros(NC, dtype=mesh.ftype)
        
        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        self.uI = np.zeros(NE, dtype=mesh.ftype) 
        self.uI[isYDEdge] = pde.velocity_x(bc[isYDEdge])
        self.uI[isXDEdge] = pde.velocity_y(bc[isXDEdge]) 
        self.pI = pde.pressure(pc)

        self.ph[0] = self.pI[0]

        I, = np.nonzero(isBDEdge & isYDEdge)
        J, = np.nonzero(isBDEdge & isXDEdge)
        self.uh[I] = self.uI[I]
        self.uh[J] = self.uI[J]

    def get_nonlinear_coef(self):
        mesh = self.mesh
        uh = self.uh
        uI = self.uI

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

        C[flag] = 1/4*(np.sqrt(uh[flag]**2+uh[P1]**2)+np.sqrt(uh[flag]**2+uh[D1]**2)\
                +np.sqrt(uh[flag]**2+uh[P2]**2)+np.sqrt(uh[flag]**2+uh[D2]**2))


        flag = ~isBDEdge & isXDEdge
        L = edge2cell[flag, 0]
        R = edge2cell[flag, 1]
        P1 = cell2edge[L, 3]
        D1 = cell2edge[L, 1]
        P2 = cell2edge[R, 3]
        D2 = cell2edge[R, 1]
        pi= np.pi
        C[flag] = 1/4*(np.sqrt(uh[flag]**2+uh[P1]**2)+np.sqrt(uh[flag]**2+uh[D1]**2)\
                +np.sqrt(uh[flag]**2+uh[P2]**2)+np.sqrt(uh[flag]**2+uh[D2]**2))
       
        C = mu/k + rho*beta*C


        return C 

    def get_left_matrix(self):
        mesh = self.mesh
        pde = self.pde
        ftype = mesh.ftype
        itype = mesh.itype

        C = self.get_nonlinear_coef()

        hx = mesh.hx
        hy = mesh.hy
        ny = mesh.ds.ny
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        cell2edge = mesh.ds.cell_to_edge()

        flag0 = mesh.ds.boundary_cell_flag(0)
        flag1 = mesh.ds.boundary_cell_flag(1)
        flag2 = mesh.ds.boundary_cell_flag(2)
        flag3 = mesh.ds.boundary_cell_flag(3)

        idx0, = np.nonzero(~flag0)
        idx2, = np.nonzero(~flag2)
        idx3, = np.nonzero(~flag3)
        idx1, = np.nonzero(~flag1)
        data1 = 1/C[cell2edge[idx3, 3]]/hx**2
        G = coo_matrix((-data1,(idx1,idx3)),shape = (NC, NC), dtype=ftype)
        G += coo_matrix((data1,(idx1,idx1)),shape = (NC, NC),dtype=ftype)

        data2 = 1/C[cell2edge[idx2, 2]]/hy**2
        G += coo_matrix((-data2,(idx0,idx2)),shape = (NC,NC),dtype=ftype)
        G += coo_matrix((data2,(idx0,idx0)),shape = (NC,NC),dtype=ftype)

        data3 = 1/C[cell2edge[idx0, 0]]/hy**2
        G += coo_matrix((-data3,(idx2,idx0)),shape = (NC,NC),dtype=ftype)
        G += coo_matrix((data3,(idx2,idx2)),shape = (NC,NC),dtype=ftype)

        data4 = 1/C[cell2edge[idx1, 1]]/hx**2
        G += coo_matrix((-data4,(idx3,idx1)),shape = (NC,NC),dtype=ftype)
        G += coo_matrix((data4,(idx3,idx3)),shape = (NC,NC),dtype=ftype)
        G = G.tocsr()

#        print('A',A)
#        print(flag0)
#        print(flag3)
#        print(idx3)
#        print(idx4)
#        print(idx1)
#        print(idx2)
        return G

    def get_right_vector(self):
        mesh = self.mesh
        pde = self.pde
        itype = mesh.itype
        ftype = mesh.ftype
        uh = self.uh

        hx = mesh.hx
        hy = mesh.hy

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        flag0 = mesh.ds.boundary_cell_flag(0)
        flag1 = mesh.ds.boundary_cell_flag(1)
        flag2 = mesh.ds.boundary_cell_flag(2)
        flag3 = mesh.ds.boundary_cell_flag(3)
        flag = mesh.ds.boundary_cell_flag()

        cell2edge = mesh.ds.cell_to_edge()
        isYDEdge = mesh.ds.y_direction_edge_flag()

        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')
       
        fx = pde.source2(bc[:sum(isYDEdge)])
        fy = pde.source3(bc[sum(isYDEdge):])
        f = np.r_[fx,fy]
        g = pde.source1(pc)
#        print('f',f)
#        print('g',g)

        C = self.get_nonlinear_coef()
        s = np.zeros(NC, dtype=ftype)

        idx, = np.nonzero(flag2 & flag3)
        s[idx] = g[idx] - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               + uh[cell2edge[idx, 3]]/hx\
               - uh[cell2edge[idx, 2]]/hy

        idx, = np.nonzero(flag0 & flag1)
        s[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               - uh[cell2edge[idx, 1]]/hx\
               + uh[cell2edge[idx, 0]]/hy

        idx, = np.nonzero(flag2 & flag1)
        s[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - uh[cell2edge[idx, 1]]/hx\
               - uh[cell2edge[idx, 2]]/hy

        idx, = np.nonzero(flag3 & ~flag0 & ~flag2)
#        print('idx',idx)
#        print('cell2edge',cell2edge[idx,:])
        s[idx] = g[idx] - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               + uh[cell2edge[idx, 3]]/hx
        

        idx, = np.nonzero(flag1 & ~flag0 & ~flag2)
        s[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - uh[cell2edge[idx, 1]]/hx

        idx, = np.nonzero(flag0 & ~flag1 & ~flag3)
        s[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               + uh[cell2edge[idx, 0]]/hy

        idx, = np.nonzero(flag2 & ~flag1 & ~flag3)
        s[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - uh[cell2edge[idx, 2]]/hy

        idx, = np.nonzero(~flag)
        s[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]

        #right

        return s

    def get_p_cell2cell(self):
        mesh = self.mesh
        ph0 = self.ph0
        cell2cell = mesh.ds.cell_to_cell()
        cell2edge = mesh.ds.cell_to_edge()
        p = ph0[cell2cell]

        idx = mesh.ds.boundary_cell_index(0)
        p[idx, 0] = - hy*f[cell2edge[idx, 0]] + ph1[idx] \
                    + hy*C[cell2edge[idx, 0]]*self.uh[cell2edge[idx, 0]]
        # right boundary cell
        idx = mesh.ds.boundary_cell_index(1)
        p[idx, 1] =   hx*f[cell2edge[idx, 1]] + ph1[idx] \
                    - hx*C[cell2edge[idx, 1]]*self.uh[cell2edge[idx, 1]]
        # up boundary cell
        idx = mesh.ds.boundary_cell_index(2)
        p[idx, 2] =   hy*f[cell2edge[idx, 2]] + ph1[idx] \
                    - hy*C[cell2edge[idx, 2]]*self.uh[cell2edge[idx, 2]]
        # left boundary cell
        idx = mesh.ds.boundary_cell_index(3)
        p[idx, 3] = - hx*f[cell2edge[idx, 3]] + ph1[idx] \
                    + hx*C[cell2edge[idx, 3]]*self.uh[cell2edge[idx, 3]]

        pass
        
    


    def solve(self):
        mesh = self.mesh
        pde = self.pde
        itype = mesh.itype
        ftype = mesh.ftype

        hx = mesh.hx
        hy = mesh.hy
        nx = mesh.ds.nx
        ny = mesh.ds.ny

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        isBDEdge = mesh.ds.boundary_edge_flag()
        cell2edge = mesh.ds.cell_to_edge()
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

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

        I = np.arange(NC, dtype=itype)
        data = np.ones(NC, dtype=ftype)
        A21 = coo_matrix((data/mesh.hx, (I, cell2edge[:, 1])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-data/mesh.hx, (I, cell2edge[:, 3])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((data/mesh.hy, (I, cell2edge[:, 2])), shape=(NC, NE), dtype=ftype)
        A21 += coo_matrix((-data/mesh.hy, (I, cell2edge[:, 0])), shape=(NC, NE), dtype=ftype)
        A21 = A21.tocsr()

        fx = pde.source2(bc[:sum(isYDEdge)])
        fy = pde.source3(bc[sum(isYDEdge):])
        f = np.r_[fx,fy]
        g = pde.source1(pc)
        tol = 1e-6
        rp = 1
        ru = 1
        e = 1
        count = 0
        iterMax = 2000

        uh1 = np.zeros((NE,), dtype=ftype)
        ph1 = np.zeros((NC,), dtype=ftype)
        ph1[0] = self.pI[0]
        C = self.get_nonlinear_coef()
    
        pI = self.pI
        uI = self.uI
        r = np.zeros((2,iterMax),dtype=ftype)
        while e > tol and count < iterMax:

            G = self.get_left_matrix()
            s = self.get_right_vector()
            bdIdx = np.zeros((G.shape[0],), dtype=itype)
            bdIdx[0] = 1
            Tbd = spdiags(bdIdx, 0, G.shape[0], G.shape[1])
            T = spdiags(1-bdIdx, 0, G.shape[0], G.shape[1])
            AD = T@G@T + Tbd
            s[0] = self.pI[0]

            #solve
            ph1[1:NC] = spsolve(G[1:NC,1:], s[1:NC])
            
#            A1 = AD.toarray()
#            D = spdiags(np.diag(A1),0,NC,NC)
#            R = A1 - D
#            s1 = np.dot(R,self.ph)
#            s2 = np.zeros(NC,dtype=ftype)
#            s2[:] = s1[:,:]
#            snew = s - s2
#            ph2 = spsolve(D,snew)

            cell2cell = mesh.ds.cell_to_cell()
            edge2cell = mesh.ds.edge_to_cell()
            p = ph1[cell2cell]
            # bottom boundary cell
            idx = mesh.ds.boundary_cell_index(0)
            p[idx, 0] = - hy*f[cell2edge[idx, 0]] + ph1[idx] \
                        + hy*C[cell2edge[idx, 0]]*self.uh[cell2edge[idx, 0]]
            # right boundary cell
            idx = mesh.ds.boundary_cell_index(1)
            p[idx, 1] =   hx*f[cell2edge[idx, 1]] + ph1[idx] \
                        - hx*C[cell2edge[idx, 1]]*self.uh[cell2edge[idx, 1]]
            # up boundary cell
            idx = mesh.ds.boundary_cell_index(2)
            p[idx, 2] =   hy*f[cell2edge[idx, 2]] + ph1[idx] \
                        - hy*C[cell2edge[idx, 2]]*self.uh[cell2edge[idx, 2]]
            # left boundary cell
            idx = mesh.ds.boundary_cell_index(3)
            p[idx, 3] = - hx*f[cell2edge[idx, 3]] + ph1[idx] \
                        + hx*C[cell2edge[idx, 3]]*self.uh[cell2edge[idx, 3]]

#            print('p',p)

            w0 = (f[cell2edge[:, 0]] - (ph1 - p[:, 0])/hy)/C[cell2edge[:, 0]]
            w1 = (f[cell2edge[:, 1]] - (p[:, 1] - ph1)/hx)/C[cell2edge[:, 1]]
            w2 = (f[cell2edge[:, 2]] - (p[:, 2] - ph1)/hy)/C[cell2edge[:, 2]]
            w3 = (f[cell2edge[:, 3]] - (ph1 - p[:, 3])/hx)/C[cell2edge[:, 3]]
            ep = np.sqrt(np.sum(hx*hy*(self.ph - ph1)**2))


            wu = np.r_[w3,w1[NC-ny:]]
            w0 = w0.reshape(ny,nx)
            w4 = w2.reshape(ny,nx)
            wv = np.column_stack((w0,w4[:,nx-1])).flatten()
            u = np.r_[wu,wv]

            C = self.get_nonlinear_coef()
            rp = LA.norm(g - A21*u)
#            rp = LA.norm(s - G*ph1)
            ru = LA.norm(f -C*u - A12*ph1)/LA.norm(f)
            eu = np.sqrt(np.sum(hx*hy*(self.uh-u)**2))
            uerror = LA.norm(A21*(self.uI-u))
            print('uerror',uerror)
            e = np.r_[eu,ep]
            e = np.max(e)
            print('ru',ru)
            print('rp',rp)

            self.ph[:] = ph1
            self.uh[:] = u
#            rp = LA.norm(g - (w1-w3)/hx - (w2-w0)/hy)
#            print('uh',self.uh)
            G = self.get_left_matrix()
            s = self.get_right_vector()

            r[0,count] = rp
            r[1,count] = ru

            count = count + 1

        self.uh = u
        self.ph = ph1
        print('solve |u|(implicit expression)')
        print('ph',self.ph)
        print('pI',self.pI)
#        
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

    def get_DpL2_error(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        hx = mesh.hx
        hy = mesh.hy
        nx = mesh.ds.nx
        ny = mesh.ds.ny
        ftype = mesh.ftype

        Dph = np.zeros((NC,2),dtype=ftype)
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')
        DpI = np.zeros((NC,2), dtype=mesh.ftype)

        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        isBDEdge = mesh.ds.boundary_edge_flag()
        I, = np.nonzero(~isBDEdge & isYDEdge)
        DpI[:NC-ny, 0] = self.pde.grad_pressure_x(bc[I])
        J, = np.nonzero(~isBDEdge & isXDEdge)

        Dph[NC-ny:NC,0] = np.zeros(ny,dtype=ftype)
        Dph[ny-1:NC:ny,1] = np.zeros(ny,dtype=ftype)
        DpI[NC-ny:NC,0] = np.zeros(ny,dtype=ftype)
        DpI[ny-1:NC:ny,1] = np.zeros(ny,dtype=ftype)

        Dph[:NC-ny,0] = (self.ph[ny:] - self.ph[:NC-ny])/hx
        
        m = np.arange(NC)
        m = m.reshape(ny,nx)
        n1 = m[:,1:].flatten()
        n2 = m[:,:ny-1].flatten()
        Dph[n2,1] = (self.ph[n1] - self.ph[n2])/hy
        DpI[n2,1] = self.pde.grad_pressure_y(bc[J])

        DpeL2 = np.sqrt(np.sum(hx*hy*(Dph[:] - DpI[:])**2))

        return DpeL2

