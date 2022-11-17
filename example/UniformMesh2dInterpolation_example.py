import numpy as np
import sys
import matplotlib.pyplot as plt

from fealpy.mesh.UniformMesh2d import UniformMesh2dFunction
from fealpy.mesh import UniformMesh2d, MeshFactory, QuadrangleMesh, TriangleMesh
import time

def ff(x):
    return (x[..., 0])**2 + (x[..., 1])**2

def ff(x):
    return np.sin(np.pi*x[..., 0])*np.sin(np.pi*x[..., 1]) 

n = int(sys.argv[1]) if len(sys.argv)>1 else 40

NP = 5
x = np.zeros([NP**2, 2], dtype=np.float_)
for i in range(NP):
    for j in range(NP):
        x[i*NP+j, 0] = i*(1.0/(NP-1));                                     
        x[i*NP+j, 1] = j*(1.0/(NP-1));                                     
        if((i > 0) & (i < NP-1) & (j > 0) & (j < NP-1)):                              
            x[i*NP+j, 0] += 0.000000001*(np.random.rand()-0.5)
            x[i*NP+j, 1] += 0.000000001*(np.random.rand()-0.5)

## 随机点
y = ff(x)

# 生成插值函数
f = UniformMesh2dFunction.from_sample_points(x, y, nx=n, ny=n)
print(x)
print(f.f)

# 计算误差
print(f.f[0, 0])
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
#plt.show()


