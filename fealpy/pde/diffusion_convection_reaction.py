
import numpy as np
from numpy.typing import NDArray
from fealpy.decorator import cartesian, barycentric
from typing import Callable, Sequence
import sympy as sp

class NonConservativeDCRPDEModel2d:
    """
	Equation:
        - \\nabla\\cdot(d \\nabla u) + b\\cdot\\nabla u + c u = f in \Omega
    """
    def __init__(self, 
            u='cos(pi*x)*cos(pi*y)', 
            A=1,
            b=[10, 10],
            c=1):
        x, y = sp.symbols('x, y')
        u = sp.sympify(u)

        du_dx = u.diff(x)
        du_dy = u.diff(y)


        A = sp.sympify(A)
        c = sp.sympify(c)

        b0 = sp.sympify(b[0])
        b1 = sp.sympify(b[1])

        s = -A*du_dx.diff(x) - A*du_dy.diff(y) + b0*du_dx + b1*du_dy + c*u 

        self._solution = sp.lambdify((x, y), u, "numpy")
        self._source = sp.lambdify((x, y), s, "numpy") 
        self._du_dx = sp.lambdify((x, y), du_dx, "numpy")
        self._du_dy = sp.lambdify((x, y), du_dy, "numpy")
        self._b0 = sp.lambdify((x, y), b0, "numpy")
        self._b1 = sp.lambdify((x, y), b1, "numpy")
        self._A = sp.lambdify((x, y), A, "numpy")
        self._c = sp.lambdify((x, y), c, "numpy")

    def domain(self):
        """
        @brief 模型定义域
        """
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def solution(self, p):
        """ 
        @brief 真解
        """
        x = p[..., 0]
        y = p[..., 1]
        return self._solution(x, y) 

    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        return self._source(x, y) 

    @cartesian
    def gradient(self, p):
        """ 
        @brief 真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        
        val[..., 0] = self._du_dx(x, y) 
        val[..., 1] = self._du_dy(x, y) 
        return val

    @cartesian
    def diffusion_coefficient(self, p):
        """
        @brief 扩散系数
        """
        x = p[..., 0]
        y = p[..., 1]
        return self._A(x, y) 

    @cartesian
    def convection_coefficient(self, p):
        """
        @brief 对流系数
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self._b0(x, y)
        val[..., 1] = self._b1(x, y)
        return val 

    @cartesian
    def reaction_coefficient(self, p):
        """
        @brief 反应系数
        """
        x = p[..., 0]
        y = p[..., 1]
        return self._c(x, y)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)


class HemkerDCRModel2d:
    def __init__(self, A=1.0, b=(1.0, 0.0)):
        self.A = A 
        self.b = b

    def domain(self):
        from fealpy.geometry import BoxWithCircleHolesDomain 
        from fealpy.geometry import huniform
        return BoxWithCircleHolesDomain(
                box=[-3.0, 9.0, -3.0, 3.0],
                circles=[(0.0, 0.0, 1.0)], fh=huniform)

    @cartesian
    def solution(self, p):
        """ 
        @brief 真解
        """
        return 0.0

    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        return 0.0

    @cartesian
    def gradient(self, p):
        """ 
        @brief 真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        return val

    @cartesian
    def diffusion_coefficient(self, p):
        """
        @brief 扩散系数
        """
        return self.A

    @cartesian
    def convection_coefficient(self, p):
        """
        @brief 对流系数
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.b[0] 
        val[..., 1] = self.b[1]
        return val 

    @cartesian
    def dirichlet(self, p):
        """
        @brief  边界条件
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(x.shape, dtype=np.float64)
        flag = x**2 + y**2 - 1 < 0.0 
        val[flag] = 1
        return val 

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定的边界点是否在 Dirichlet 边界内部
        """
        x = p[..., 0]
        y = p[..., 1]
        return (x**2 + y**2 - 1 < 0.0) | np.isclose(x, -3.0, atol=1e-12)

class HemkerDCRModelWithBoxHole2d:
    def __init__(self, A=1.0, b=(1.0, 0.0)):
        from fealpy.geometry import BoxWithBoxHolesDomain 
        from fealpy.geometry import huniform
        from fealpy.geometry import drectangle
        self.A = A 
        self.b = b
        self._domain = BoxWithBoxHolesDomain(
                box=[-3.0, 9.0, -3.0, 3.0],
                boxs=[(-1.0, 1.0, -1.0, 1.0)], fh=huniform)
        self.fd = lambda p: drectangle(p, [-1.0, 1.0, -1.0, 1.0])

    def domain(self):
        return self._domain

    @cartesian
    def solution(self, p):
        """ 
        @brief 真解
        """
        return 0.0

    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        return 0.0

    @cartesian
    def gradient(self, p):
        """ 
        @brief 真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        return val

    @cartesian
    def diffusion_coefficient(self, p):
        """
        @brief 扩散系数
        """
        return self.A

    @cartesian
    def convection_coefficient(self, p):
        """
        @brief 对流系数
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.b[0] 
        val[..., 1] = self.b[1]
        return val 

    @cartesian
    def dirichlet(self, p):
        """
        @brief  边界条件
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(x.shape, dtype=np.float64)
        flag = np.isclose(self.fd(p), 0.0, atol=1e-12)
        val[flag] = 1
        return val 

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定的边界点是否在 Dirichlet 边界内部
        """
        x = p[..., 0]
        y = p[..., 1]
        flag0 = np.isclose(self.fd(p), 0.0, atol=1e-12)
        flag1 = np.isclose(x, -3.0, atol=1e-12)
        return flag0 | flag1
    

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

