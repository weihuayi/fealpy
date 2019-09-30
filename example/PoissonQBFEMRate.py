#!/usr/bin/env python3
# 
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# 导入 Poisson 有限元模型
from fealpy.fem.PoissonQBFEMModel import PoissonQBFEMModel
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.pde.poisson_2d import CosCosData as PDE

n = 3
maxit = 4
pde = PDE() # 创建 pde 模型

# 误差类型与误差存储数组
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

# 自由度数组
Ndof = np.zeros(maxit, dtype=np.int)

# 创建初始网格对象 
mesh = pde.init_mesh(n, meshtype='quad')

for i in range(maxit):
    fem = PoissonQBFEMModel(pde, mesh, q=3) # 创建 Poisson 有限元模型
    ls = fem.solve() # 求解
    Ndof[i] = fem.space.number_of_global_dofs() # 获得空间自由度个数
    errorMatrix[0, i] = fem.L2_error() # 计算 L2 误差
    errorMatrix[1, i] = fem.H1_semi_error() # 计算 H1 误差
    if i < maxit - 1:
        mesh.uniform_refine() # 一致加密网格

# 显示误差
show_error_table(Ndof, errorType, errorMatrix)
# 可视化误差收敛阶
showmultirate(plt, 0, Ndof, errorMatrix, errorType)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

