import sys 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory
from fealpy.mesh.PolygonMesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.tools.show import showmultirate, show_error_table
from IPDGModel_poly import IPDGModel
from fealpy.pde.poisson_2d import PolynomialData as PDE

p = 2        #int(sys.argv[2]) # 有限元空间的次数
maxit =4     #int(sys.argv[4]) # 迭代加密的次数
pde = PDE()  # 创建 pde 模型
beta = 200
alpha = -1

# 误差类型与误差存储数组
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
# 自由度数组
NDof = np.zeros(maxit, dtype=np.int)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    mesh = pde.init_mesh(n=i+3, meshtype='tri') 
    mesh = PolygonMesh.from_mesh(mesh)
    # mesh.add_plot(plt, cellcolor='w')
    # mesh = pde.init_mesh(n=2**(i+3), meshtype='poly') 
    # mesh.add_plot(plt, cellcolor='w')
    space = ScaledMonomialSpace2d(mesh,p)
    NDof[i] = space.number_of_global_dofs()
    
    fem = IPDGModel(pde, mesh, p) # 创建 Poisson 有限元模型
    fem.solve(beta,alpha) # 求解
    uh = fem.uh
    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)
    
# 显示误差
show_error_table(NDof, errorType, errorMatrix)
# 可视化误差收敛阶
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=12)

plt.show()


