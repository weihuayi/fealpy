import numba
import numpy as np

class NodeSetKernelSpace:
    
    def __init__(self, mesh, ker='Quintic', H=1):
        """
        @brief 
        @param[in] mesh NodeSet 类型的网格
        @param[in] ker str kernel 函数计算的字符串
        """
        self.mesh = mesh
        self.ker = ker
        self.H = H
         
    #@numba.jit(nopython=True)
    def kernel(self, r):
        choose = self.ker
        d = np.sqrt(np.sum(r**2, axis=-1))
        H = self.H
        q = d/H
        if choose == 'Quintic':
            val = 7 * (1-q/2)**4 * (2*q+1) / (4*np.pi*H**2)        
        return val
    
    '''
    @numba.jit(nopython=True)
    def Quintic_kernel(self, r):
        d = np.sqrt(np.sum(r**2, axis=-1))
        q = d/H
        val = 7 * (1-q/2)**4 * (2*q+1) / (4*np.pi*H**2)
        return val
    '''

    def grad_kernel(self, r):
        d = np.sqrt(np.sum(r**2))
        H = self.H
        q = d/H
        choose = self.ker

        if choose == 'Quintic':
            val = -35/(4*np.pi*H**3) * q * (1-q/2)**3
            val *= r
            if d==0:
                val = 0
            else :
                val /= d
        return val

    def value(self, u, points):
        """
        @brief 
        @param[in] u 定义在节点上的物理量的值
        @param[in] points 需要求值的点 

        @return 物理量在 points 处的函数值
        """
        pass


    def grad_value(self, u, points):
        """
        @brief 
        @param[in] u 定义在节点上的物理量的值
        @param[in] points 需要求值的点 

        @return 物理量在 points 处的梯度值
        """
        pass

