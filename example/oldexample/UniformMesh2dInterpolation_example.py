import numpy as np
import sys
import matplotlib.pyplot as plt

from fealpy.mesh.UniformMesh2d import UniformMesh2dFunction
from fealpy.mesh import UniformMesh2d, MeshFactory, QuadrangleMesh, TriangleMesh
import time

def ff(x):
    return (2*x[..., 0]-1)**2 + (2*x[..., 1]-1)**2

n = int(sys.argv[1]) if len(sys.argv)>1 else 40


## 随机点
x = np.random.random((25, 2))
y = ff(x)

# 生成插值函数
f = UniformMesh2dFunction.from_sample_points(x, y, nx=n, ny=n)

# 计算误差
error = np.max(np.abs(f.f[:-1, :-1] - ff(f.mesh.entity('node')[:-1, :-1])))
print('最大值误差：', error)

## 画图
mesh = f.mesh

node = mesh.entity('node').reshape(-1, 2)
node0 = np.c_[node, f.f.reshape(-1)[:, None]]
node1 = np.c_[node, ff(node)[:, None]]
cell0 = mesh.entity('cell')[:, [0, 2, 1]]
cell1 = mesh.entity('cell')[:, [1, 2, 3]]
cell = np.r_[cell0, cell1]

mesh0 = TriangleMesh(node0, cell)
mesh1 = TriangleMesh(node1, cell)
mesh0.to_vtk(fname='000.vtu')
mesh1.to_vtk(fname='111.vtu')

fig = plt.figure()
axes = fig.gca()
axes.scatter(x[:, 0], x[:, 1])
plt.show()


