
import numpy as np
from fealpy.decorator import cartesian, barycentric
import sympy as sp

class NonConservativeDCRPDEModel2d:
    """
	Equation:
        - \\nabla\\cdot(d \\nabla u) + b\\cdot\\nabla u + c u = f in \Omega
    """
    def __init__(self, 
            u='cos(pi*x)*cos(pi*y)', 
            d=1,
            b=[10, 10],
            c=1):
        x, y = sp.symbols('x, y')
        u = sp.sympify(u)

        du_dx = u.diff(x)
        du_dy = u.diff(y)


        d = sp.sympify(d)
        c = sp.sympify(c)

        b0 = sp.sympify(b[0])
        b1 = sp.sympify(b[1])

        s = -d*du_dx.diff(x) - d*du_dy.diff(y) + b0*du_dx + b1*du_dy + c*u 

        self._solution = sp.lambdify((x, y), u, "numpy")
        self._source = sp.lambdify((x, y), s, "numpy") 
        self._du_dx = sp.lambdify((x, y), du_dx, "numpy")
        self._du_dy = sp.lambdify((x, y), du_dy, "numpy")
        self._b0 = sp.lambdify((x, y), b0, "numpy")
        self._b1 = sp.lambdify((x, y), b1, "numpy")
        self._d = sp.lambdify((x, y), d, "numpy")
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
        @brief 对流系数
        """
        x = p[..., 0]
        y = p[..., 1]
        return self._d(x, y) 

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
        @brief 对流系数
        """
        x = p[..., 0]
        y = p[..., 1]
        return self._c(x, y)


    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

