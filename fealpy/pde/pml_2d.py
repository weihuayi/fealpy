
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Sequence
import sympy as sp

from fealpy.decorator import cartesian


class PMLPDEModel2d:
    def __init__(self, 
                 levelset:Callable[[NDArray], NDArray],
                 domain:Sequence[float],
                 u_inc:str,
                 A:float,
                 k:float,
                 d:Sequence[float], 
                 refractive_index:Sequence[float], 
                 absortion_constant:float, 
                 lx:float, 
                 ly:float):  
        
        x, y, n= sp.symbols('x, y, n')
        replacement_dict = {'d_0': str(d[0]), 'd_1': str(d[1]), 'k': str(k)}
        for variable, value in replacement_dict.items():
            u_inc = u_inc.replace(variable, value)
        
        u_inc = sp.sympify(u_inc)
        
        A = sp.sympify(A)
        K = sp.sympify(k)
       
        du_inc_dx = u_inc.diff(x)
        du_inc_dy = u_inc.diff(y)
  
        s = A * du_inc_dx.diff(x) + A * du_inc_dy.diff(y) + K**2 * n * u_inc
        
        self.absortion_constant = absortion_constant 
        self.lx = lx 
        self.ly = ly
        self.k = k  
        self.d = d
        self.levelset = levelset
        self.domain = domain
        self.refractive_index = refractive_index
        self._source = sp.lambdify((x, y, n), s, "numpy")
 
    @cartesian
    def e_x(self, p):
        domain = self.domain
        d_x = self.lx
        sigma_0 = self.absortion_constant

        a1 = domain[0] + d_x
        b1 = domain[1] - d_x

        x = p[..., 0]
        sigma_x = np.zeros_like(x, dtype=np.complex128)

        idx_1 = (x > domain[0]) & (x < a1)
        idx_2 = (x > a1) & (x < b1)
        idx_3 = (x > b1) & (x < domain[1])

        sigma_x[idx_1] = sigma_0 * (((x[idx_1] - a1) / d_x) ** 2)
        sigma_x[idx_2] = 0.0
        sigma_x[idx_3] = sigma_0 * (((x[idx_3] - b1) / d_x) ** 2)
        val = 1.0 + sigma_x * 1j
        return val
    
    @cartesian
    def e_x_d_x(self, p):
        domain = self.domain
        d_x = self.lx
        sigma_0 = self.absortion_constant

        a1 = domain[0] + d_x
        b1 = domain[1] - d_x

        x = p[..., 0]
        sigma_x_d_x = np.zeros_like(x, dtype=np.complex128)

        idx_1 = (x > domain[0]) & (x < a1)
        idx_2 = (x > a1) & (x < b1)
        idx_3 = (x > b1) & (x < domain[1])

        sigma_x_d_x[idx_1] = sigma_0 * 2 * (1/d_x) * ((x[idx_1] - a1) / d_x)
        sigma_x_d_x[idx_2] = 0.0
        sigma_x_d_x[idx_3] = sigma_0 * 2 * (1/d_x) * ((x[idx_3] - b1) / d_x)
        val = sigma_x_d_x * 1j
        return val

    @cartesian
    def e_y(self, p):
        domain = self.domain
        d_y = self.ly
        sigma_0 = self.absortion_constant

        a2 = domain[2] + d_y
        b2 = domain[3] - d_y

        y = p[..., 1]
        sigma_y = np.zeros_like(y, dtype=np.complex128)

        idx_1 = (y > domain[2]) & (y < a2)
        idx_2 = (y > a2) & (y < b2)
        idx_3 = (y > b2) & (y < domain[3])

        sigma_y[idx_1] = sigma_0 * (((y[idx_1] - a2) / d_y) ** 2)
        sigma_y[idx_2] = 0.0
        sigma_y[idx_3] = sigma_0 * (((y[idx_3] - b2) / d_y) ** 2)
        val = 1.0 + sigma_y * 1j
        return val

    @cartesian    
    def e_y_d_y(self, p):
        domain = self.domain
        d_y = self.ly
        sigma_0 = self.absortion_constant

        a2 = domain[2] + d_y
        b2 = domain[3] - d_y

        y = p[..., 1]
        sigma_y_d_y = np.zeros_like(y, dtype=np.complex128)

        idx_1 = (y > domain[2]) & (y < a2)
        idx_2 = (y > a2) & (y < b2)
        idx_3 = (y > b2) & (y < domain[3])

        sigma_y_d_y[idx_1] = sigma_0 * 2 * (1/d_y) * ((y[idx_1] - a2) / d_y)
        sigma_y_d_y[idx_2] = 0.0
        sigma_y_d_y[idx_3] = sigma_0 * 2 * (1/d_y) * ((y[idx_3] - b2) / d_y)
        val = sigma_y_d_y * 1j
        return val

    @cartesian 
    def n_(self, p):
        origin = p.shape[:-1]
        p = p.reshape(-1, 2)
        x = p[..., 0]
        if np.all(np.isin(self.levelset(p), [True, False])):
            flag = self.levelset(p)
        else:
            flag = self.levelset(p)< 0. # (NC, )
        n = np.empty((x.shape[-1], ), dtype=np.complex128)
        n[flag] = self.refractive_index[1]
        n[~flag] = self.refractive_index[0]
        n = n.reshape(origin)
        return n
    
    @cartesian
    def source(self, p):
        """
        @brief 源项
        """
        n = self.n_(p)
        x = p[..., 0]
        y = p[..., 1]
        val = self._source(x, y, n)
        return val

    @cartesian
    def diffusion_coefficient(self, p):
        """
        @brief 扩散系数
        """
        x = p[..., 0]
        GD = 2
        e_x = self.e_x(p)
        e_y = self.e_y(p)
        val = np.zeros((p.shape[0], x.shape[-1], GD, GD), dtype=np.complex128)
        val[:, :, 0, 0] = e_y/e_x
        val[:, :, 1, 1] = e_x/e_y
        return val

    @cartesian
    def convection_coefficient(self, p):
        """
        @brief 对流系数
        """
        val = np.zeros(p.shape, dtype=np.complex128)
        val[..., 0] = self.e_y(p) * (-1/self.e_x(p)**2) * self.e_x_d_x(p)
        val[..., 1] = self.e_x(p) * (-1/self.e_y(p)**2) * self.e_y_d_y(p)
        return -val

    @cartesian
    def reaction_coefficient(self, p):
        """
        @brief 反应系数
        """
        x = p[..., 0]
        k = self.k
        n = self.n_(p)
        val = np.zeros(x.shape, dtype=np.complex128) 
        val[:] = self.e_x(p) * self.e_y(p) * n * k**2
        return -val

    @cartesian
    def dirichlet(self, p):
        x = p[..., 0]
        return np.zeros(x.shape, dtype=np.complex128)

