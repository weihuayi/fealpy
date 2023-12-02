import numpy as np
from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.vem import PoissonACVEMSolver

from fealpy.mesh import PolygonMesh
import matplotlib.pyplot as plt

meshfilepath = './data/mesh/'# 需要创建相应的文件夹
equfilepath = './data/equ/' # # 需要创建相应的文件夹

pde = LShapeRSinData()
mesh = pde.init_mesh(n=1,meshtype='quad')
solver = PoissonACVEMSolver(pde,mesh)
solver.adaptive_solve(maxit=4,save_data=True,meshfilepath=meshfilepath,equfilepath=equfilepath)

# 读取保存的数据
data = np.load(meshfilepath+"mesh1.npz")
node = data['node']
cell = data['cell']
cellLocation = data['cellLocation']

# 重新生成网格，数据可以使用
mesh = PolygonMesh(node,cell,cellLocation)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
