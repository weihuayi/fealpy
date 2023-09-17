
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
        @brief 对流系数
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
        @brief 对流系数
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
        @brief 对流系数
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
        @brief 对流系数
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

