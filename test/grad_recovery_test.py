#!/usr/bin/env python3
# 
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# 导入 Poisson 有限元模型
from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.tools.show import showmultirate, show_error_table

from fealpy.mesh.adaptive_tools import mark


# 问题维数
d = int(sys.argv[1])

if d == 1:
    from fealpy.pde.poisson_1d import CosData as PDE
elif d==2:
    from fealpy.pde.poisson_2d import LShapeRSinData as PDE
elif d==3:
    from fealpy.pde.poisson_3d import LShapeRSinData as PDE

p = int(sys.argv[2]) # 有限元空间的次数
n = int(sys.argv[3]) # 初始网格的加密次数
maxit = int(sys.argv[3])
theta = int(sys.argv[4])
pde = PDE() # 创建 pde 模型
mesh = pde.init_mesh(n)
errorType = ['$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$']


Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    print('step:', i)
    fem = PoissonFEMModel(pde, mesh, p, q=p+2) # 创建 Poisson 有限元模型
    fem.solve() # 求解
    uh = fem.uh
    space = uh.space
    Ndof[i] = space.mesh.number_of_nodes()
    errorMatrix[0, i] = fem.L2_error() # 计算 L2 误差
    errorMatrix[1, i] = fem.H1_semi_error() # 计算 H1 误差
    
    rguh = space.grad_recovery(uh)
    eta = fem.recover_estimate(rguh)
    markedCell = mark(eta,theta=theta,method='MAX')
    if i < maxit - 1:
        markedCell = mark(eta,theta=theta,method='MAX')
        mesh.bisect(markedCell)

# 可视化误差收敛阶
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# 可视化网格
mesh.add_plot(plt, cellcolor='w')