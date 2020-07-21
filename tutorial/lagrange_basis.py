
import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace


p = int(sys.argv[1])

mf = MeshFactory()

box = [0, 1, 0, 1]
mesh = mf.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

# 打印网格信息
NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NF = mesh.number_of_faces()
NC = mesh.number_of_cells()
print("网格中节点、边和单元的个数分别为：", NN, NE, NC)


print('创建拉格朗日有限元空间...')
space = LagrangeFiniteElementSpace(mesh, p=p)
ldof = space.number_of_local_dofs()
gdof = space.number_of_global_dofs()

print('拉格朗日空间的次数为：', p)
print('每个单元上的局部自由度个数：', ldof)
print('每个单元上的全局自由度个数', gdof)

print('计算空间基函数在每个单元重心坐标点处的值...')
bc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)# (3, 3)
ps = mesh.bc_to_point(bc) # (NQ, NC, 2) 

phi = space.basis(bc)
gphi = space.grad_basis(bc)

print('重心坐标数组：', bc)
print('bc.shape:', bc.shape)
print('phi.shape:', phi.shape)
print('gphi.shape:', gphi.shape)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
