import numpy as np
from scipy.sparse import diags
from scipy.sparse import diags, lil_matrix
from scipy.sparse import vstack
from scipy.sparse.linalg import spsolve


class NSMacSolver():
    def __init__(self, umesh, vmesh, pmesh):
        self.umesh = umesh
        self.vmesh = vmesh
        self.pmesh = pmesh
        

    def grad_ux(self):
        mesh = self.umesh
        dx = mesh.h[0]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol

        result = diags([1, -1],[Nrow, -Nrow],(N,N), format='csr')

        return result/(2*dx)

    def grad_uy(self):
        mesh = self.umesh
        dy = mesh.h[1]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([1, -1],[1, -1],(N,N), format='lil')

        index = np.arange(0, N, Nrow)
        result[index, index] = 2
        result[index, index+1] = 2/3
        result[index[1:], index[1:]-1] = 0

        index = np.arange(Nrow-1, N, Nrow)
        result[index, index] = -2
        result[index, index-1] = -2/3
        result[index[:-1], index[:-1]+1] = 0
        return result/(2*dy)
    
    def Tuv(self):
        mesh  = self.umesh
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = np.zeros((N,N))
        index = np.arange(N-Nrow)
        result[index,index+index//Nrow] = 1
        result[index,index+index//Nrow+1] = 1
        result[index,index+index//Nrow-Nrow-1] = 1
        result[index,index+index//Nrow-Nrow] = 1
        return result/4
    
    def laplace_u(self):
        mesh = self.umesh
        dx,dy = mesh.h
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([-4, 1, 1, 1, 1],[0, 1, -1, Nrow, -Nrow],(N,N), format='lil')

        index = np.arange(0,N, Nrow)
        result[index, index] = -6
        result[index[2:]-1, index[2:]-1] = -6
        result[index[1:], index[1:]-1] = 0
        result[index[2:]-1, index[2:]] = 0
        result[index, index+1] = 4/3
        result[index[2:]-1, index[2:]-2] = 4/3

        return result/(dx*dy)
    
    def grand_uxp(self):
        mesh = self.pmesh
        dx = mesh.h[0]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([1, -1],[0, -Nrow],(N,N), format='lil')
        A = lil_matrix((Nrow, N))
        result1 = vstack([result, A], format='lil')   
        return result1/dx
    
    def source_Fx(self, pde ,t):
        mesh = self.umesh
        nodes = mesh.entity('node')
        source = pde.source_F(nodes,t) 
        return source

