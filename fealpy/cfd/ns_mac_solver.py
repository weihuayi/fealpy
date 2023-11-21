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
    
    def grad_vx(self):
        mesh = self.vmesh
        dx = mesh.h[0]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol

        result = diags([1, -1],[Nrow, -Nrow],(N,N), format='csr')
        index = np.arange(0,N)
        result[index[:Nrow],index[:Nrow]] = 2
        result[index[:Nrow],index[:Nrow]+Nrow] = 2/3
        result[index[-Nrow:],index[-Nrow:]] = -2
        result[index[-Nrow:],index[-Nrow:]-Nrow] = -2/3
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
    
    def grad_vy(self):
        mesh = self.vmesh
        dy = mesh.h[1]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([1, -1],[1, -1],(N,N), format='lil')
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
    
    def Tvu(self):
        mesh = self.vmesh
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = np.zeros((N,N))
        arr = np.arange(0,N)
        split_array = np.array_split(arr,Ncol)
        lists = [sub_array.tolist() for sub_array in split_array]
        num = len(lists)
        for i in range(num):
            i_array = np.ones_like(lists[i])
            result[lists[i],lists[i]-i*i_array] = 1
            result[lists[i],lists[i]-i*i_array-1] = 1
        for i in range(num-1):
            i_array = np.ones_like(lists[i])
            result[lists[i],lists[i]-i*i_array+Ncol*i_array] = 1
            result[lists[i],lists[i]-i*i_array-1+Ncol*i_array] = 1
        index = lists[-1][:Nrow-1]
        N_array = np.ones_like(index)
        result[index,index] = 1
        result[index,index+N_array] = 1
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
    
    def laplace_v(self):
        mesh = self.vmesh
        dx,dy = mesh.h
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([-1, 1, 1, 1, 1],[0, 1, -1, Nrow, -Nrow],(N,N), format='lil')

        index = np.arange(0,N)
        result[index[:Nrow],index[:Nrow]] = -6
        result[index[-Nrow:],index[-Nrow:]] = -6
        result[index[:Nrow],index[:Nrow]+Nrow] = 4/3
        result[index[-Nrow:],index[-Nrow:]-Nrow] = 4/3

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
    
    def grand_vyp(self):
        mesh = self.pmesh
        dx = mesh.h[0]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        N1 = Nrow*Ncol + Nrow
        result = diags([0],[0],(N1,N), format='lil')
        arr = np.arange(0,N1)
        split_array = np.array_split(arr,Nrow)
        lists = [sub_array.tolist() for sub_array in split_array]
        num = len(lists)
        for i in range(num-1):
            i_array= np.ones_like(lists[i])
            result[lists[i],lists[i]-i*i_array] = 1
            result[lists[i],lists[i]-i*i_array-1] = -1
        index = lists[-1][:Nrow]
        num_array = np.ones_like(index)
        result[index,index-(num-1)*num_array] = 1
        result[index,index-(num-1)*num_array-1] = -1
        return result/dx
    
    def source_Fx(self, pde ,t):
        mesh = self.umesh
        nodes = mesh.entity('node')
        source = pde.source_F(nodes,t) 
        return source
    
    def grad_pux(self):
        mesh = self.umesh
        dx = mesh.h[0]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([1,1],[0,Nrow],(N-Nrow,N), format='lil')
        return result/dx
    
    def grad_pvy(self):
        mesh = self.vmesh
        dy = mesh.h[1]
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([1,1],[0,1],(N-Ncol,N), format='lil')
        return result/dy

    def laplaplace_phi(self):
        mesh = self.pmesh
        dx,dy = mesh.h
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        result = diags([-4,1,1,1,1],[0,1,-1,Nrow,-Nrow],(N,N), format='lil')
        index = np.arange(0,N)
        index0 = np.where(index%Nrow ==0)
        index1 = np.ones_like(index0)
        result[index0,index0-index1] = 0
        result[index0-index1,index0] = 0
        result[index[:Nrow],index[:Nrow]] = -3
        result[index[-Nrow:],index[-Nrow:]] = -3
        result[index0,index0] = -3
        result[index0-index1,index0-index1] = -3
        result[0,0] = -2
        result[Nrow-1,Nrow-1] = -2
        result[N-1,N-1] = -2
        result[N-Nrow,N-Nrow] = -2
        return result/(dx*dy)