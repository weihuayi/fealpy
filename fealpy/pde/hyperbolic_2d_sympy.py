from sympy import symbols, diff, lambdify, sympify, pi
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

class Hyperbolic2dData:
    def __init__(self, u_expr, a=1.0, D=[0, 1, 0, 1], T=[0, 1]):
        self._domain = D
        self._duration = T
        self.a = a  # 统一速度系数

        # 定义符号变量
        x, y, t = symbols('x y t')
        self.x, self.y, self.t = x, y, t

        # 用户输入的解析解表达式
        self.u_expr = sympify(u_expr)
        self.u = lambdify([x, y, t], self.u_expr)

        # 计算右端项 f = ∂u/∂t + a(∂u/∂x + ∂u/∂y)
        du_dt = diff(self.u_expr, t)
        du_dx = diff(self.u_expr, x)
        du_dy = diff(self.u_expr, y)
        self.f_expr = du_dt + a * (du_dx + du_dy)
        self.f = lambdify([x, y, t], self.f_expr)

    def domain(self):
        return self._domain

    def duration(self):
        return self._duration

    def solution(self, p, t):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        t = bm.array(t)
        return self.u(x, y, t)

    def init_solution(self, p):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        return self.u(x, y, self._duration[0])

    @cartesian
    def source(self, p, t):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        return self.f(x, y, t)

    def dirichlet(self, p, t):
        return self.solution(p, t)
