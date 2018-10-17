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
        C[flag] = 1/4*(np.sqrt(uh[flag]**2+uh[P1]**2)+np.sqrt(uh[flag]**2+uh[D1]**2)\
                +np.sqrt(uh[flag]**2+uh[P2]**2)+np.sqrt(uh[flag]**2+uh[D2]**2))

        C = mu/k + rho*beta*C

        return C 

    def solve(self):
        mesh = self.mesh
        pde = self.pde
        itype = mesh.itype
        ftype = mesh.ftype

        hx = mesh.hx
        hy = mesh.hy

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        # find edge
        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()

        edge2cell = mesh.ds.edge_to_cell()
        cell2edge = mesh.ds.cell_to_edge()
        I, = np.nonzero(~isBDEdge & isYDEdge)
        Lx = edge2cell[I, 0]
        Rx = edge2cell[I, 1]

        J, = np.nonzero(~isBDEdge & isXDEdge)
        Ly = edge2cell[J, 0]
        Ry = edge2cell[J, 1]

        cell2cell = mesh.ds.cell_to_cell()

        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        uh1 = np.zeros(NE, dtype=ftype)
        ph1 = np.zeros(NC, dtype=ftype)
        ph1[0] = pde.pressure(pc[0])
        
        fx = pde.source2(bc[:sum(isYDEdge)])
        fy = pde.source3(bc[sum(isYDEdge):])
        f = np.r_[fx,fy]
        g = pde.source1(pc)

        tol = 1e-6
        ru = 1
        rp = 1
        count = 0
        iterMax = 2000

        ph = self.ph 
        uh = self.uh
        while ru+rp > tol and count < iterMax:
            C = self.get_nonlinear_coef()
            p = ph[cell2cell]

            # bottom boundary cell
            idx = mesh.ds.boundary_cell_index(0)
            p[idx, 0] = - hy*f[cell2edge[idx, 0]] + ph[idx] \
                        + hy*C[cell2edge[idx, 0]]*uh[cell2edge[idx, 0]] 
            # right boundary cell
            idx = mesh.ds.boundary_cell_index(1)
            p[idx, 1] =   hx*f[cell2edge[idx, 1]] + ph[idx] \
                        - hx*C[cell2edge[idx, 1]]*uh[cell2edge[idx, 1]]
            # up boundary cell
            idx = mesh.ds.boundary_cell_index(2)
            p[idx, 2] =   hy*f[cell2edge[idx, 2]] + ph[idx] \
                        - hy*C[cell2edge[idx, 2]]*uh[cell2edge[idx, 2]]
            # left boundary cell
            idx = mesh.ds.boundary_cell_index(3)
            p[idx, 3] = - hx*f[cell2edge[idx, 3]] + ph[idx] \
                        + hx*C[cell2edge[idx, 3]]*uh[cell2edge[idx, 3]] 

            ph1[1:]  = (g[1:] - f[cell2edge[1:, 1]]/hx/C[cell2edge[1:, 1]]\
                             + f[cell2edge[1:, 3]]/hx/C[cell2edge[1:, 3]]\
                             - f[cell2edge[1:, 2]]/hy/C[cell2edge[1:, 2]]\
                             + f[cell2edge[1:, 0]]/hy/C[cell2edge[1:, 0]]\
                             + p[1:, 1]/hx**2/C[cell2edge[1:, 1]]\
                             + p[1:, 3]/hx**2/C[cell2edge[1:, 3]]\
                             + p[1:, 2]/hy**2/C[cell2edge[1:, 2]]\
                             + p[1:, 0]/hy**2/C[cell2edge[1:, 0]])\
                             /(1/hx**2/C[cell2edge[1:, 1]]\
                             + 1/hx**2/C[cell2edge[1:, 3]]\
                             + 1/hy**2/C[cell2edge[1:, 0]]\
                             + 1/hy**2/C[cell2edge[1:, 2]])

            uh1[I] = (f[I] - (ph1[Rx]-ph1[Lx])/hx)/C[I]
            uh1[J] = (f[J] - (ph1[Ly]-ph1[Ry])/hy)/C[J]


            ru = np.sqrt(np.sum(hx*hy*(uh1-uh)**2))
            rp = np.sqrt(np.sum(hx*hy*(ph1-ph)**2))
#            print('rp:',rp)
#            print('ru',ru)
#            print('ph0:',self.ph0)
#            print('uh0:',self.uh0)
#            print('uh1:',uh1)
#            print('ph1:',p)

            self.ph[:] = ph1
            self.uh[:] = uh1

#            print('ph0:',self.ph0)
#            print('uh0:',self.uh0)

            count = count + 1

        self.uh = uh1
        self.ph = ph1
#        print('uh:',self.uh)
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

