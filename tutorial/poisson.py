
"""
亲爱的小伙伴们，大家好！今天我来给大家演示如何基于 FEALPy， 来实现 Lagrange
有限元元求解 Poisson 方程的数值实验程序。注意这里所用的界面是 Jupyter Notebook

这里考虑一个真解为 cos(pi*x)cos(pi*y) 的 Poisson 方程, 带有纯 Dirichlet
边界条件。
"""

"""
首先，我们导入多维数组模块 numpy 和画图模块 matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

"""
接着，我们导入真解已知的 PDE 模型类，并建立相应的 PDE 模型对象 
"""
from fealpy.pde.poisson_2d import CosCosData
pde = CosCosData()


"""
再导入我们的网格工厂模块，用于生成网格
"""
from fealpy.mesh import MeshFactory as MF

mesh = MF.boxmesh2d(pde.domain(), nx=10, ny=10, meshtype='tri')
mesh.add_plot(plt)


"""
下面再导入我们的 Lagrane 有限元空间
"""
from fealpy.functionspace import LagrangeFiniteElementSpace

space = LagrangeFiniteElementSpace(mesh, p=1)
uh = space.function()
A = space.stiff_matrix()
F = space.source_vector(pde.source)


from fealpy.boundarycondition import DirichletBC

bc = DirichletBC(space, pde.dirichlet) 
A, F = bc.apply(A, F, uh)


from scipy.sparse.linalg import spsolve

uh[:] = spsolve(A, F)
L2error = space.integralalg.error(pde.solution, uh)
H1error = space.integralalg.error(pde.gradient, uh.grad_value)

print("L2 error: ", L2error)
print("H1 error: ", H1error)

uh.add_plot(plt, cmap="rainbow")

plt.show()

