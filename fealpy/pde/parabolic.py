from ..backend import backend_manager as bm
from ..decorator import cartesian
from sympy import *


class ParabolicData: 
    def __init__(self,u, var_list, t, D=None, T=[0, 1]):
        """
        Initialize the parabolic equation data.

        Parameters:
            u (str): The analytical solution expression.
            var_list (list): List of independent variables (supports arbitrary dimensions).
            t (sympy.Symbol): The time variable.
            D (list, optional): The domain range. Defaults to [0, 1] for each variable.
            T (list, optional): The time range. Defaults to [0, 1].
        """
        self._domain = D if D is not None else [0, 1] * len(var_list) 
        self._duration = T
        self._var_list = symbols(var_list)  # 动态生成变量
        u_expr = sympify(u)
        self.u = lambdify([*self._var_list, t], u_expr)
        self.f = lambdify([*self._var_list, t], diff(u_expr, t, 1) - 
                          sum(diff(u_expr, var, 2) for var in self._var_list))

        self.gradients = {
            str(var): lambdify([*self._var_list, t], diff(u_expr, var, 1))
            for var in self._var_list
        }
        self.t = t
        
    def domain(self):
        """
        Get the domain of the parabolic equation.

        Returns:
            list: The domain range.
        """
        return self._domain

    def duration(self):
        """
        Get the time range of the parabolic equation.

        Returns:
            list: The time range.
        """
        return self._duration 

    @cartesian
    def solution(self, p, t):
        """
        Compute the solution value.

        Parameters:
            p (array): The coordinates.
            t (float): The time.

        Returns:
            array: The solution value at the given coordinates and time.
        """
        variables = [p[..., i] for i in range(p.shape[1])]
        return self.u(*variables, t)

    @cartesian
    def init_solution(self, p):
        """
        Compute the initial solution.

        Parameters:
            p (array): The coordinates.

        Returns:
            array: The initial solution value at the given coordinates.
        """
        return self.solution(p, self._duration[0])
    
    @cartesian
    def source(self, p, t):
        """
        Compute the source term.

        Parameters:
            p (array): The coordinates.
            t (float): The time.

        Returns:
            array: The source term value at the given coordinates and time.
        """
        variables = [p[..., i] for i in range(p.shape[1])]
        return self.f(*variables, t)
    
    @cartesian
    def gradient(self, p, t):
        """
        Compute the gradient of the solution.

        Parameters:
            p (array): The coordinates.
            t (float): The time.

        Returns:
            array: The gradient values, shaped (NN, GD), where NN is the number 
            of points and GD is the geometric dimension.
        """
        variables = [p[..., i] for i in range(p.shape[1])]

        # 计算每个维度的梯度值
        gradients = [
            grad_func(*variables, t) for grad_func in self.gradients.values()
        ]
        # 将梯度值组合成 (NN, GD) 的数组
        return bm.stack(gradients, axis=1)
    
    @cartesian
    def dirichlet(self, p, t):
        return self.solution(p, t)