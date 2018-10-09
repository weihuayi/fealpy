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

        return C 

    def solve(self):
        mesh = self.mesh
        pde = self.pde

        hx = mesh.hx
        hy = mesh.hy
        Nx = int(1/hx)
        Ny = int(1/hy)
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        itype = mesh.itype
        ftype = mesh.ftype

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
        print('idx',idx)
        cell = np.arange(NC)
        vy1 = cell2edge[cell, 0] # e_{i,0}
        ux2 = cell2edge[cell, 1] # e_{i,1}
        vy2 = cell2edge[cell, 2] # e_{i,2}
        ux1 = cell2edge[cell, 3] # e_{i,3}#correct

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
        iterMax = 500
        while ru+rp > tol and count < iterMax:
            C = self.get_nonlinear_coef()
            fx = pde.source2(bc[:sum(isYDEdge)])
            fy = pde.source3(bc[sum(isYDEdge):])
            f = np.r_[fx,fy]
            g = pde.source1(pc)

            #boundary cell of the vertex angle
            ph1[Ny-1] = (g[Ny-1] - (hx*f[ux2[Ny-1]]-self.ph0[2*Ny-1])/C[ux2[Ny-1]]/hx**2\
                    + (hy*f[vy1[Ny-1]]+self.ph0[Ny-1])/C[vy1[Ny-1]]/hy**2)\
                    / (1/C[ux2[Ny-1]]/hx**2 + 1/C[vy1[Ny-1]]/hy**2)
            ph1[Ny*(Nx-1)] = (g[Ny*(Nx-1)]+(hx*f[ux1[Ny*(Nx-1)]]+self.ph0[Ny*(Nx-2)])/C[ux1[Ny*(Nx-1)]]/hx**2\
                    -(hy*f[vy2[Ny*(Nx-1)]]-self.ph0[Ny*(Nx-1)+1])/C[vy2[Ny*(Nx-1)]]/hy**2)\
                    /(1/C[ux1[Ny*(Nx-1)]]/hx**2+1/C[vy2[Ny*(Nx-1)]]/hy**2)
            ph1[Ny*Nx-1]=(g[Ny*Nx-1]+(hx*f[ux1[Ny*Nx-1]]+self.ph0[Ny*(Nx-1)-1])/C[ux1[Ny*Nx-1]]/hx**2\
                    +(hy*f[vy1[Ny*Nx-1]]+self.ph0[Ny*Nx-2])/C[vy1[Ny*Nx-1]]/hy**2)\
                    /(1/C[ux1[Ny*Nx-1]]/hx**2+1/C[vy1[Ny*Nx-1]]/hy**2) 
            print('ph0:',self.ph0)

            #left boundary cell
            idx1 = idx[1:Ny-1]
            ph1[idx1] =(g[idx1]-(hx*f[ux2[idx1]]-self.ph0[idx1+Ny])/C[ux2[idx1]]/hx**2\
                    - (hy*f[vy2[idx1]]-self.ph0[idx1+1])/C[vy2[idx1]]/hy**2 \
                    + (hy*f[vy1[idx1]] + self.ph0[idx1-1])/C[vy1[idx1]]/hy**2)\
                    /(1/C[ux2[idx1]]/hx**2+(1/C[vy2[idx1]]+1/C[vy1[idx1]])/hy**2)

            print('ph0:',self.ph0)#????
            #right boundary cell
            idx1 = idx[len(idx)-Ny+1:len(idx)-1]

            ph1[idx1] =(g[idx1]+(hx*f[ux1[idx1]]+self.ph0[idx1-Ny])/C[ux1[idx1]]/hx**2\
                    - (hy*f[vy2[idx1]]-self.ph0[idx1+1])/C[vy2[idx1]]/hy**2 \
                    + (hy*f[vy1[idx1]] + self.ph0[idx1-1])/C[vy1[idx1]]/hy**2)\
                    /(1/C[ux1[idx1]]/hx**2+(1/C[vy2[idx1]]+1/C[vy1[idx1]])/hy**2)

            #upper boundary cell
            idx1 = idx[Ny:len(idx)-Ny:2]
            ph1[idx1]=(g[idx1]-(hx*f[ux2[idx1]]-self.ph0[idx1+Ny])/C[ux2[idx1]]/hx**2\
                    +(hx*f[ux1[idx1]] + self.ph0[idx1-Ny])/C[ux1[idx1]]/hx**2 \
                    - (hy*f[vy2[idx1]] - self.ph0[idx1+1])/C[vy2[idx1]]/hy**2)\
                    /((1/C[ux1[idx1]]+1/C[ux2[idx1]])/hx**2+1/C[vy2[idx1]]/hy**2)

            #under boundary cell
            idx1 = idx[Ny+1:len(idx)-Ny:2]

            ph1[idx1]=(g[idx1]-(hx*f[ux2[idx1]]-self.ph0[idx1+Ny])/C[ux2[idx1]]/hx**2\
                    +(hx*f[ux1[idx1]] + self.ph0[idx1-Ny])/C[ux1[idx1]]/hx**2 \
                    + (hy*f[vy1[idx1]] + self.ph0[idx1-1])/C[vy1[idx1]]/hy**2)\
                    /((1/C[ux1[idx1]]+1/C[ux2[idx1]])/hx**2+1/C[vy1[idx1]]/hy**2)

            # internal cell
            idx1, = np.nonzero(~isBDCell)
            ph1[idx1] = (g[idx1]-(hx*f[ux2[idx1]]-self.ph0[idx1+Ny])/C[ux2[idx1]]/hx**2\
                    +(hx*f[ux1[idx1]] + self.ph0[idx1-Ny])/C[ux1[idx1]]/hx**2 \
                    - (hy*f[vy2[idx1]]-self.ph0[idx1+1])/C[vy2[idx1]]/hy**2 \
                    + (hy*f[vy1[idx1]] + self.ph0[idx1-1])/C[vy1[idx1]]/hy**2)\
                    /((1/C[ux1[idx1]]+1/C[ux2[idx1]])/hx**2+(1/C[vy2[idx1]]+1/C[vy1[idx1]])/hy**2)

            uh1[I] = (f[I] - (ph1[Rx]-ph1[Lx])/hx)/C[I]
            uh1[J] = (f[J] - (ph1[Ly]-ph1[Ry])/hy)/C[J]


            ru = np.sqrt(np.sum(hx*hy*(uh1-self.uh0)**2))
            rp = np.sqrt(np.sum(hx*hy*(ph1-self.ph0)**2))
            print('rp:',rp)
            print('ru',ru)

#            if LA.norm(f) == 0:
#                ru = LA.norm(f - A11*u1 - A12*p1)
#            else:
#                ru = LA.norm(f - A11*u1 - A12*p1)/LA.norm(f)
#            if LA.norm(g) == 0:
#                rp = LA.norm(g - A21*u1)
#            else:
#                rp = LA.norm(g - A21*u1)/LA.norm(g)
            print('ph0:',self.ph0)
#            print('uh0:',self.uh0)
#            print('uh1:',uh1)
#            print('ph1:',ph1)
            ph0 = ph1
            uh0 = uh1

            self.ph0 = ph0
            self.uh0 = uh0

            print('ph0:',self.ph0)
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

