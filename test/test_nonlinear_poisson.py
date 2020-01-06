import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

n = 4 # 初始网格加密次数
p = 1 # 有限元空间次数
pde = CosCosData() # 准备 pde 模型
mesh = pde.init_mesh(n=4) # 生成初始网格
space = LagrangeFiniteElementSpace(mesh, 1) # 构建有限元空间
A = space.stiff_matrix() # 构造刚度矩阵
b = space.source_vector(pde.source) # 构造载荷向量
# 处理边界条件
bc = DirichletBC(space, pde.dirichlet)
AD, b = bc.apply(A, b)
# 代数方程求解
uh = space.function()
uh[:] = spsolve(AD, b)
# 计算误差
error = space.integralalg.L2_error(pde.solution, uh)
print(error)
