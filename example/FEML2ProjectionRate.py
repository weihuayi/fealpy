import numpy as np
import matplotlib.pyplot as plt
import sys

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.tools.show import showmultirate, show_error_table

d = int(sys.argv[1])# 问题维数
if d == 1:
    from fealpy.pde.poisson_1d import CosData as PDE
elif d==2:
    from fealpy.pde.poisson_2d import CosCosData as PDE

p = int(sys.argv[2]) # 有限元空间的次数
n = int(sys.argv[3]) # 初始网格的加密次数
maxit = int(sys.argv[4]) # 迭代加密的次数

pde = PDE()
mesh = pde.init_mesh(n)
errorType = ['$|| U - P_u||_0$', '$||\\nabla U - \\nabla P_u||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

# 自由度数组
Ndof = np.zeros(maxit, dtype=np.int)
for i in range(maxit):
    space = LagrangeFiniteElementSpace(mesh, p, q=p+2)
    uh = space.projection(pde.solution)

    Ndof[i] = space.number_of_global_dofs()
    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, 
            uh.grad_value)
    if i < maxit - 1:
        mesh.uniform_refine()
# 显示误差
show_error_table(Ndof, errorType, errorMatrix)
# 可视化误差收敛阶
showmultirate(plt, 0, Ndof, errorMatrix, errorType)


















