import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import csc_matrix, csr_matrix, spdiags, eye
from scipy.sparse.linalg import spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..femmodel import doperator 
from .integral_alg import IntegralAlg
from .doperator  import mass_matrix, grad_recovery_matrix


class CahnHilliardRFEMModel():
    def __init__(self, pde, n, tau, q):
        self.pde = pde 
        self.mesh = pde.space_mesh(n) 
        self.timemesh, self.tau = self.pde.time_mesh(tau)
        self.femspace = LagrangeFiniteElementSpace(self.mesh, 1) 

        self.uh = self.femspace.function(dim=len(self.timemesh))
        self.uh[:, 0] = self.femspace.interpolation(pde.initdata)

        self.area = self.mesh.entity_measure('cell')

        self.integrator = self.mesh.integrator(q)
        self.integralalg = IntegralAlg(self.integrator, self.mesh, self.area)

        self.A, self.B, self.gradphi = grad_recovery_matrix(self.femspace)
        self.M = doperator.mass_matrix(self.femspace, self.integrator, self.area)
        self.K = self.get_stiff_matrix()  
        self.D = self.M + self.pde.epsilon**2 * self.tau * self.K

        self.current = 0

    def get_stiff_matrix(self):
        mesh = self.mesh
        area = self.area
        
        gradphi = self.gradphi 
        A = self.A
        B = self.B
        NC = mesh.number_of_cells() 
        NN = mesh.number_of_nodes() 

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:,0]==edge2cell[:,1])
        bdEdge = edge[isBdEdge]
        cellIdx = edge2cell[isBdEdge, [0]]
        
        # construct the unit outward normal on the boundary
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (node[bdEdge[:,1],] - node[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape(-1, 1)

        # 计算梯度的恢复矩阵
        I = np.einsum('ij, k->ijk',  cell, np.ones(3))
        J = I.swapaxes(-1, -2)
        
        val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 0], gradphi[:, :, 0])
        P = csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
        
        val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 0], gradphi[:, :, 1])
        Q = csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
        
        val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 1], gradphi[:, :, 1])
        S = csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
        
        K = A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q.transpose()@A+B.transpose()@S@B 

        # 边界上两个方向导数相乘的积分
        I = np.einsum('ij, k->ijk', bdEdge, np.ones(2))
        J = I.swapaxes(-1, -2)
        val = np.array([(1/3, 1/6), (1/6, 1/3)])
        val0 = np.einsum('i, jk->ijk', n[:, 0]*n[:, 0]/h, val)        
        P = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

        val0 = np.einsum('i, jk->ijk', n[:, 0]*n[:, 1]/h, val)
        Q = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

        val0 = np.einsum('i, jk->ijk', n[:, 1]*n[:, 1]/h, val)
        S = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))
        
        K += A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q@A + B.transpose()@S@B

        # 中间的边界上的两项
        I = np.einsum('ij, k->ijk', bdEdge, np.ones(3))
        J = np.einsum('ij, k->ikj', cell[cellIdx], np.ones(2)) 
        val0 = 0.5*h.reshape(-1, 1)*n[:, [0]]*gradphi[cellIdx, :, 0]  
        val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
        P0 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

        val0 = 0.5*h.reshape(-1, 1)*n[:, [0]]*gradphi[cellIdx, :, 1]  
        val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
        Q0 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

        val0 = 0.5*h.reshape(-1, 1)*n[:, [1]]*gradphi[cellIdx, :, 0]  
        val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
        P1 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

        val0 = 0.5*h.reshape(-1, 1)*n[:, [1]]*gradphi[cellIdx, :, 1]  
        val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
        Q1 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

        M = A.transpose()@P0@A + A.transpose()@Q0@B + B.transpose()@P1@A + B.transpose()@Q1@B

        K -= (M + M.transpose())
        return K 

    def get_right_vector(self, i):
        uh = self.uh.index(i)
        b =  self.get_non_linear_vector(uh)
        return self.M@uh  - self.tau*b


    def get_non_linear_vector(self, uh):
        bcs, ws = self.integrator.quadpts, self.integrator.weights
        gradphi = self.femspace.grad_basis(bcs)

        uval = uh.value(bcs)
        guval = uh.grad_value(bcs)
        #print(gradphi.shape)
        fval = (3*uval[..., np.newaxis]**2 - 1)*guval
        bb = np.einsum('i, ikm, ikjm, k->kj', ws, fval, gradphi, self.area)
        cell2dof = self.femspace.cell_to_dof()
        gdof = self.femspace.number_of_global_dofs()
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b

    def solve(self):
        timemesh = self.timemesh 
        tau = self.tau
        N = len(timemesh)
        D = self.D
        for i in range(N-1):
            t = timemesh[i]
            b = self.get_right_vector(i)
            self.uh[:, i+1] =  spsolve(D, b)
            print(self.uh[:, i+1])
            self.show_soultion()
            

    def step(self):
        D = self.D
        b = self.get_right_vector(self.current)
        self.uh[:, self.current + 1] = spsolve(D, b)
        self.current += 1
       

    def show_soultion(self):
        mesh = self.mesh
        timemesh = self.timemesh 
        cell = mesh.entity('cell')
        N = len(timemesh)
        print(N)
        x = mesh.node[:, 0]
        y = mesh.node[:, 1]
        for i in range(0,N-1,1):
            fig = plt.figure()
            fig.set_facecolor('white')
            axes = fig.gca(projection='3d')
            axes.plot_trisurf(x, y, cell, self.uh[:,i], cmap=plt.cm.jet, lw=0.0)
            plt.savefig('./results/cahnHilliard'+ str(i)+'.png')
    
    
