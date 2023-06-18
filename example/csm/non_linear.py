"""
多变量 N-R 方法主程序（收敛例子）
"""
import numpy as np

from fealpy.opt import Problem

class TwoNonlinearSpingsProblem(Problem):
    def __init__(self):
        x0 = np.zeros(2, dtype=np.float64)
        super().__init__(x0, self.energy)

    def enery(self, x):
        k0, k1 = self.springs_stiffness(x)

    def springs_stiffness(self, x):
        return 50 + 500*x[0], 100 + 200*x[1]

    def stiffness_matrix(self, x):
        return 





from fealpy.solver.nonlinear_solver import NonlinearSolver


# 非线性弹簧刚度矩阵
def my_calculate_P(u):
    return np.array([300*u[0]**2+400*u[0]*u[1]-200*u[1]**2+150*u[0]-100*u[1],
                     200*u[0]**2-400*u[0]*u[1]+200*u[1]**2-100*u[0]+100*u[1]])

# jacobian 矩阵（切线刚度矩阵）
def my_calculate_Kt(u):
    return np.array([[600*u[0]+400*u[1]+150, 400*(u[0]-u[1])-100],
                     [400*(u[0]-u[1])-100, 400*u[1]-400*u[0]+100]])


tol = 1.0e-5 # 容差
max_iter = 20 # 最大迭代次数
u = np.array([0, 0]) # 初始解
f = np.array([0, 100]) # 外力

solver = NonlinearSolver(tol, max_iter)
result = solver.newton_raphson_bivariate(u, f, my_calculate_P, my_calculate_Kt)
print("最终结果：", result)
