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

        return C 

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

        BDx, = np.nonzero(isBDEdge & isYDEdge)
        BDy, = np.nonzero(isBDEdge & isXDEdge)

        isBDCell = mesh.ds.boundary_cell_flag()
        idx, = np.nonzero(isBDCell)
        cell = np.arange(NC)
        vy1 = cell2edge[cell, 0] # e_{i,0}
        ux2 = cell2edge[cell, 1] # e_{i,1}
        vy2 = cell2edge[cell, 2] # e_{i,2}
        ux1 = cell2edge[cell, 3] # e_{i,3}#correct

        cell2cell = mesh.ds.cell_to_cell()

        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        self.ph0[0] = pde.pressure(pc[0])
        uh1 = np.zeros(NE, dtype=ftype)
        ph1 = np.zeros(NC, dtype=ftype)
        ph1[0] = pde.pressure(pc[0])

        tol = 1e-6
        ru = 1
        rp = 1
        count = 0
        iterMax = 2000
        while ru+rp > tol and count < iterMax:
            C = self.get_nonlinear_coef()
            fx = pde.source2(bc[:sum(isYDEdge)])
            fy = pde.source3(bc[sum(isYDEdge):])
            f = np.r_[fx,fy]
            g = pde.source1(pc)

            p = np.zeros((NC,4),dtype=ftype)
            idx1 = cell2cell[:NC:ny, 0]
            a = np.arange(NC).reshape(ny,nx)
            idx2 = a[:, 1:].flatten()
            idx3 = a[:,:nx-1].flatten()
            p[idx1, 0] = hy*C[vy1[idx1]]*pde.velocity_y(bc[cell2edge[idx1, 0]]) - hy*f[vy1[idx1]] + self.ph0[idx1]
            p[idx2, 0] = self.ph0[idx3]

            idx1 = cell2cell[NC-ny:NC, 1]
            idx2 = a[:ny-1, :].flatten()
            idx3 = a[1:, :].flatten()
            p[idx1, 1] = hx*f[ux2[idx1]] - hx*C[ux2[idx1]]*pde.velocity_x(bc[cell2edge[idx1, 1]]) + self.ph0[idx1]
            p[idx2, 1] = self.ph0[idx3]

            idx1 = cell2cell[ny-1:NC:ny, 2]
            idx2 = a[:, :nx-1].flatten()
            idx3 = a[:, 1:].flatten()
            p[idx1, 2] = hy*f[vy2[idx1]] - hy*C[vy2[idx1]]*pde.velocity_y(bc[cell2edge[idx1, 2]]) + self.ph0[idx1]
            p[idx2, 2] = self.ph0[idx3]
#            print('idx1',idx1)
#            print('idx2',idx2)
#            print('idx3',idx3)
#            print('vy1[idx1]',vy2[idx1])
#            print('cell2edge[idx1,2]',cell2edge[idx1,2])
#            print('f',f[vy2[idx1]])
#            print('ff',hx*f[vy2[idx1]])
#            print('C',C[vy2[idx1]])
#            print('pde',pde.velocity_y(bc[cell2edge[idx1,2]]))
#            print('CC',hx*C[vy2[idx1]]*pde.velocity_y(bc[cell2edge[idx1, 2]]))
#            print('ph0',self.ph0)
#            print('p[idx2]',p[idx2,0])
#            print('p[idx1]',p[idx1,3])

            idx1 = cell2cell[:ny, 3]
            idx2 = a[1:, :].flatten()
            idx3 = a[:ny-1,:].flatten()
            p[idx1, 3] = hx*C[ux1[idx1]]*pde.velocity_x(bc[cell2edge[idx1, 3]]) - hx*f[ux1[idx1]] + self.ph0[idx1]
            p[idx2, 3] = self.ph0[idx3]
            print('p',p)


            ph1[1:] = (g[1:] - f[ux2[1:]]/hx/C[ux2[1:]] + f[ux1[1:]]/hx/C[ux1[1:]]- f[vy2[1:]]/hy/C[vy2[1:]] + f[vy1[1:]]/hy/C[vy1[1:]]\
                            + p[1:, 1]/hx**2/C[ux2[1:]]\
                            + p[1:, 3]/hx**2/C[ux1[1:]]\
                            + p[1:, 0]/hy**2/C[vy1[1:]]\
                            + p[1:, 2]/hy**2/C[vy2[1:]])\
                            /(1/C[ux1[1:]]/hx**2 + 1/C[vy1[1:]]/hy**2 + 1/C[ux2[1:]]/hx**2 + 1/C[vy2[1:]]/hy**2) 


            uh1[I] = (f[I] - (ph1[Rx]-ph1[Lx])/hx)/C[I]
            uh1[J] = (f[J] - (ph1[Ly]-ph1[Ry])/hy)/C[J]


            ru = np.sqrt(np.sum(hx*hy*(uh1-self.uh0)**2))
            rp = np.sqrt(np.sum(hx*hy*(ph1-self.ph0)**2))
            print('rp:',rp)
#            print('ru',ru)
#            print('ph0:',self.ph0)
#            print('uh0:',self.uh0)
#            print('uh1:',uh1)
#            print('ph1:',ph1)

            self.ph0[:] = ph1
            self.uh0[:] = uh1

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

