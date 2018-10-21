import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        C = np.zeros(NE, dtype=mesh.ftype)
        edge2cell = mesh.ds.edge_to_cell()
        cell2edge = mesh.ds.cell_to_edge()

        bc = mesh.entity_barycenter('edge')

        flag = isBDEdge & isYDEdge
        C[flag] = self.pde.velocity_x(bc[flag])

        flag = isBDEdge & isXDEdge
        C[flag] = self.pde.velocity_y(bc[flag])

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

        idx1, = np.nonzero(~flag3)
        idx2, = np.nonzero(~flag1)
        data1 = 1/C[cell2edge[idx1, 1]]/hx**2
        A = coo_matrix((-data1,(idx2,idx1)),shape = (NC, NC), dtype=ftype)
        A += coo_matrix((data1,(idx2,idx2)),shape = (NC, NC),dtype=ftype)

        idx3, = np.nonzero(~flag0)
        idx4, = np.nonzero(~flag2)
        data2 = 1/C[cell2edge[idx4, 0]]/hy**2
        A += coo_matrix((-data2,(idx3,idx4)),shape = (NC,NC),dtype=ftype)
        A += coo_matrix((data2,(idx3,idx3)),shape = (NC,NC),dtype=ftype)

        data3 = 1/C[cell2edge[idx3, 2]]/hy**2
        A += coo_matrix((-data3,(idx4,idx3)),shape = (NC,NC),dtype=ftype)
        A += coo_matrix((data3,(idx4,idx4)),shape = (NC,NC),dtype=ftype)

        data4 = 1/C[cell2edge[idx2, 3]]/hx**2
        A += coo_matrix((-data4,(idx1,idx2)),shape = (NC,NC),dtype=ftype)
        A += coo_matrix((data4,(idx1,idx1)),shape = (NC,NC),dtype=ftype)
        A = -A

#        print('A',A)
#        print(flag0)
#        print(flag3)
#        print(idx3)
#        print(idx4)
#        print(idx1)
#        print(idx2)
        return A

    def get_right_vector(self):
        mesh = self.mesh
        pde = self.pde
        itype = mesh.itype
        ftype = mesh.ftype
        uI = self.uI

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

        C = self.get_nonlinear_coef()
        b = np.zeros(NC, dtype=ftype)

        idx, = np.nonzero(flag0 & flag3)
        b[idx] = g[idx] - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               + uI[cell2edge[idx, 3]]/hx\
               + uI[cell2edge[idx, 0]]/hy

        idx, = np.nonzero(flag2 & flag3)
        b[idx] = g[idx] - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               + uI[cell2edge[idx, 3]]/hx\
               - uI[cell2edge[idx, 2]]/hy

        idx, = np.nonzero(flag0 & flag1)
        b[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               - uI[cell2edge[idx, 1]]/hx\
               + uI[cell2edge[idx, 0]]/hy

        idx, = np.nonzero(flag2 & flag1)
        b[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - uI[cell2edge[idx, 1]]/hx\
               - uI[cell2edge[idx, 2]]/hy

        idx, = np.nonzero(flag3 & ~flag0 & ~flag2)
        b[idx] = g[idx] - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               + uI[cell2edge[idx, 3]]/hx
        
        idx, = np.nonzero(flag1 & ~flag0 & ~flag2)
        b[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - uI[cell2edge[idx, 1]]/hx

        idx, = np.nonzero(flag0 & ~flag1 & ~flag3)
        b[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]\
               + uI[cell2edge[idx, 0]]/hy

        idx, = np.nonzero(flag2 & ~flag1 & ~flag3)
        b[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - uI[cell2edge[idx, 2]]/hy

        idx, = np.nonzero(~flag)
        b[idx] = g[idx] + f[cell2edge[idx, 3]]/hx/C[cell2edge[idx, 3]]\
               - f[cell2edge[idx, 1]]/hx/C[cell2edge[idx, 1]]\
               + f[cell2edge[idx, 0]]/hy/C[cell2edge[idx, 0]]\
               - f[cell2edge[idx, 2]]/hy/C[cell2edge[idx, 2]]

        return b        


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


#        print('C',C)
#        print('A',A[ny:2*ny,:])
#        print('b',b)

        isYDEdge = mesh.ds.y_direction_edge_flag()
        cell2edge = mesh.ds.cell_to_edge()
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        fx = pde.source2(bc[:sum(isYDEdge)])
        fy = pde.source3(bc[sum(isYDEdge):])
        f = np.r_[fx,fy]
        g = pde.source1(pc)
#        print('f',f)
#        print('g',g)
        tol = 1e-6
        rp = 1
        ru = 1
        count = 0
        iterMax = 2

        uh1 = np.zeros((NE,), dtype=ftype)
        ph1 = np.zeros((NC,), dtype=ftype)
    
        pI = self.pI
        uI = self.uI
        while rp+ru > tol and count < iterMax:

            C = self.get_nonlinear_coef()
            A = self.get_left_matrix()
            b = self.get_right_vector()
            bdIdx = np.zeros((A.shape[0],), dtype=itype)
            bdIdx[0] = 1
            Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
            T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
            AD = T@A@T + Tbd
            print('A',AD)
            print('b',b)
            b[0] = self.pI[0]
#            print('b',b[0])

            ph1[:] = spsolve(AD, b)

            cell2cell = mesh.ds.cell_to_cell()
            edge2cell = mesh.ds.edge_to_cell()
            p = ph1[cell2cell]
           # print('p',p)
#            isBDEdge = mesh.ds.boundary_edge_flag()
#            isYDEdge = mesh.ds.y_direction_edge_flag()
#            isXDEdge = mesh.ds.x_direction_edge_flag()
#            idx, = np.nonzero(~isBDEdge & isYDEdge)
#            uh1[idx] = (f[idx] - (ph1[edge2cell[idx, 1]]-ph1[edge2cell[idx, 0]])/hx)/C[idx] 
#            idx, = np.nonzero(~isBDEdge & isXDEdge)
#            uh1[idx] = (f[idx] - (ph1[edge2cell[idx, 0]]-ph1[edge2cell[idx, 1]])/hy)/C[idx]
#
#            rp = np.sqrt(np.sum(hx*hy*(self.ph-ph1)**2))
#            ru = np.sqrt(np.sum(hx*hy*(self.uh-uh1)**2))


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
#            rp = LA.norm(hx*hy*g - (w1-w3)*hy - (w2-w0)*hx)
#            print('rp:',rp)

            wu = np.r_[w3,w1[NC-ny:]]
            w0 = w0.reshape(ny,nx)
            w4 = w2.reshape(ny,nx)
            wv = np.column_stack((w0,w4[:,nx-1])).flatten()
            w = np.r_[wu,wv]
#            we = w-uI
#            print('we',we)
#            print('ru',ru)
#            print('ph0:',self.ph0)
#            print('w:',w)
#            print('uh1:',uh1)
#            print('ph1:',p)

            self.ph[:] = ph1
            self.uh[:] = w
            print('uh',uh1)
#            C = self.get_nonlinear_coef()
            A = self.get_left_matrix()
            b = self.get_right_vector()

#            print('ph0:',self.ph0)
#            print('uh0:',self.uh0)

            count = count + 1

        self.uh = w
        self.ph = ph1
#        
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

