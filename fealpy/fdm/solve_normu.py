import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy.linalg import norm 
from ..fem.integral_alg import IntegralAlg
from scipy.sparse.linalg import cg, inv, dsolve, spsolve

class solve_normu():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        self.uh = np.zeros(NE, dtype=mesh.ftype)
        self.ph = np.zeros(NC, dtype=mesh.ftype)
        self.uh0 = np.zeros(NE, dtype=mesh.ftype)
        self.ph0 = np.zeros(NC, dtype=mesh.ftype)

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
        pde = self.pde
        itype = mesh.itype
        ftype = mesh.ftype

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        hx = mesh.hx
        hy = mesh.hy

        C = self.get_nonlinear_coef()
        cell2edge = mesh.ds.cell_to_edge()
        edge2cell = mesh.ds.edge_to_cell()

        flag0 = mesh.ds.boundary_cell_flag(0)
        flag1 = mesh.ds.boundary_cell_flag(1)
        flag2 = mesh.ds.boundary_cell_flag(2)
        flag3 = mesh.ds.boundary_cell_flag(3)

        idx0, = np.nonzero(~flag0)
        idx1, = np.nonzero(~flag1)
        idx2, = np.nonzero(~flag2)
        idx3, = np.nonzero(~flag3)

        data0 = 1/C[cell2edge[idx0, 0]]/hy**2
        print('data0',data0)
        A = coo_matrix((-data0,(idx0, idx2)), shape=(NC,NC), dtype=ftype)
        A += coo_matrix((data0,(idx0, idx0)), shape=(NC,NC), dtype=ftype)

        data1 = 1/C[cell2edge[idx1, 1]]/hx**2
        print('data1',data1)
        A += coo_matrix((-data1,(idx1, idx3)), shape=(NC,NC), dtype=ftype)
        A += coo_matrix((data1, (idx1, idx1)), shape=(NC,NC), dtype=ftype)

        data2 = 1/C[cell2edge[idx2, 2]]/hy**2
        print('data2',data2)
        A += coo_matrix((-data2,(idx2, idx0)), shape=(NC,NC), dtype=ftype)
        A += coo_matrix((data2, (idx2, idx2)), shape=(NC,NC), dtype=ftype)

        data3 = 1/C[cell2edge[idx3, 3]]/hx**2
        print('data3',data3)
        A += coo_matrix((-data3,(idx3, idx1)), shape=(NC,NC), dtype=ftype)
        A += coo_matrix((data3,(idx3, idx3)), shape=(NC,NC), dtype=ftype)

        return A

    def get_right_vector_f(self):

        mesh = self.mesh
        pde = self.pde
        ftype = mesh.ftype

        NE = mesh.number_of_edges()
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

        f = np.zeros(NE, dtype=ftype)
        flag = ~isBDEdge & isYDEdge
        f[flag] = pde.source2(bc[flag])
        flag = ~isBDEdge & isXDEdge
        f[flag] = pde.source3(bc[flag])

        idx, = np.nonzero(isYDEdge & isBDEdge)
        val = pde.velocity_x(bc[idx])
        f[idx] = C[idx]*val #modify

        idx, = np.nonzero(isXDEdge & isBDEdge)
        val = pde.velocity_y(bc[idx])
        f[idx] = C[idx]*val

        return f


    def get_right_vector(self):
        mesh = self.mesh
        ftype = mesh.ftype
        uh0 = self.uh0

        hx = mesh.hx
        hy = mesh.hy

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        pc = mesh.entity_barycenter('cell')
        g = self.pde.source1(pc)

        C = self.get_nonlinear_coef()
        f = self.get_right_vector_f()

        s = np.zeros(NC, dtype=ftype)
        cell2edge = mesh.ds.cell_to_edge()

        s[:] = g[:] + f[cell2edge[:, 0]]/hy/C[cell2edge[:, 0]]\
                    - f[cell2edge[:, 1]]/hx/C[cell2edge[:, 1]]\
                    - f[cell2edge[:, 2]]/hy/C[cell2edge[:, 2]]\
                    + f[cell2edge[:, 3]]/hx/C[cell2edge[:, 3]]

        return s

    def solve(self):
        mesh = self.mesh
        itype = mesh.itype
        ftype = mesh.ftype

        hx = mesh.hx
        hy = mesh.hy
        nx = mesh.ds.nx
        ny = mesh.ds.ny

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()


        pc = mesh.entity_barycenter('cell')
        g = self.pde.source1(pc)

        tol = 1e-6
        rp = 1
        ru = 1
        ep = 1
        eu = 1
        count = 0
        iterMax = 2000

        r = np.zeros((2,iterMax),dtype=ftype)

        while eu+ep > tol and count < iterMax:

            C = self.get_nonlinear_coef()
            G = self.get_left_matrix()
            f = self.get_right_vector_f()
            s = self.get_right_vector()

            snew = s.copy()

            uh1 = np.zeros(NE, dtype=ftype)
            ph1 = np.zeros(NC, dtype=ftype)
            ph1[0] = self.pI[0]

            snew -= G@ph1
            snew[0] = self.pI[0]

            bdIdx = np.zeros((G.shape[0],), dtype=itype)
            bdIdx[0] = 1
            Tbd = spdiags(bdIdx, 0, G.shape[0], G.shape[1])
            T = spdiags(1-bdIdx, 0, G.shape[0], G.shape[1])
            GD = T@G@T + Tbd

            ph1[1:] = spsolve(GD[1:,1:], snew[1:])
            ph1[0] = self.pI[0]

            cell2cell = mesh.ds.cell_to_cell()
            edge2cell = mesh.ds.edge_to_cell()
            cell2edge = mesh.ds.cell_to_edge()
            p = ph1[cell2cell]

            w0 = (f[cell2edge[:, 0]] - (ph1 - p[:, 0])/hy)/C[cell2edge[:, 0]]
            w1 = (f[cell2edge[:, 1]] - (p[:, 1] - ph1)/hx)/C[cell2edge[:, 1]]
            w2 = (f[cell2edge[:, 2]] - (p[:, 2] - ph1)/hy)/C[cell2edge[:, 2]]
            w3 = (f[cell2edge[:, 3]] - (ph1 - p[:, 3])/hx)/C[cell2edge[:, 3]]

            wu = np.r_[w3,w1[NC-ny:]]
            w5 = w0.reshape(ny,nx)
            w4 = w2.reshape(ny,nx)
            wv = np.column_stack((w5,w4[:,nx-1])).flatten()
            uh1 = np.r_[wu,wv]
            eu = np.sqrt(np.sum(hx*hy*(self.uh0 - uh1)**2))
            ep = np.sqrt(np.sum(hx*hy*(self.ph0 - ph1)**2))

            self.ph0[:] = ph1
            self.uh0[:] = uh1

            if norm(s) == 0:
                rp = norm(s - G@ph1)
            else:
                rp = norm(s - G@ph1)/norm(s)

            print(rp)
            e = np.r_[eu,ep]
            e = np.max(e)

            r[0,count] = rp
            r[1,count] = ru

            count = count + 1

        self.uh[:] = uh1
        self.ph[:] = ph1

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
        print(self.ph - self.pI)
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

