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
'''    
    def grad_ux(self):
        mesh = self.mesh
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol

        result = diags([1, -1],[Nrow, -Nrow],(N,N), format='csr')

        return result

    def grad_uy(self):
        mesh = self.mesh  
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

    def Tuv(self):
        mesh  = self.mesh
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = np.zeros((N,N))
        index = np.arange(N-Nrow)
        result[index,index+index//Nrow] = 1
        result[index,index+index//Nrow+1] = 1
        result[index,index+index//Nrow-Nrow-1] = 1
        result[index,index+index//Nrow-Nrow] = 1
        return result
    
    def laplace_u(self):
        mesh = self.mesh
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

        return result
    
    def grand_uxp(self):
        mesh = self.mesh
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([1, -1],[0, -Nrow],(N,N), format='lil')
        A = lil_matrix((Nrow, N))
        result1 = vstack([result, A], format='lil')   
        return result1
    
    def source_F(self, mesh, t):
        mesh_p = mesh
        nodes_p = mesh_p.entity('node')
        num_nodes_p = nodes_p.shape[0]
        # 初始化数组，存储体积力
        result = np.zeros((num_nodes_p, 1))
        source_p = pde.source_F(nu, t) 
        result += source_p * np.ones((num_nodes_p, 1))
        return result

'''
