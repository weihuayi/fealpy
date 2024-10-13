import numpy as np
from ..decorator import cartesian

class CosData:
    def domain(self):
        return  [0.0, 1.0] 

    @cartesian
    def solution(self, p):
        """
        @brief 方程真解

        @param[in] p NDArray 
        """
        x = p[..., 0]
        val = np.cos(np.pi*x)
        return val 

    @cartesian
    def gradient(self, p):
        """
        @brief 方程真解的的导数

        """
        pi = np.pi
        val = -pi*np.sin(pi*p)
        return val

    @cartesian
    def source(self, p):
        """
        @brief 方程的源项
        """
        x = p[..., 0] # (NQ, NC)
        val = np.pi**2*np.cos(np.pi*x)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

