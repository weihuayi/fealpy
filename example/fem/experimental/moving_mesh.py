#!/usr/bin/python3
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次有限元方法求解possion方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nrefine',
        default=4, type=int,
        help='初始网格一致加密次数.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")

args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.fem import PoissonLFEMSolver, LinearElasticityLFEMSolver
from fealpy.material.elastic_material import LinearElasticMaterial



p = args.degree
n = args.nrefine

tmr = timer()
next(tmr)

pde = LShapeRSinData() 
domain = pde.domain()


mesh = pde.init_mesh(n=n, meshtype='tri') 
cell = mesh.entity('cell')
node = mesh.entity('node')

kargs = bm.context(cell)
cornerIdx = bm.array([0, 1, 3, 4, 5, 7], **kargs)
mesh.meshdata['cornerIdx'] = cornerIdx
tmr.send('网格生成时间')

s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)
uh = s0.solve()

h = bm.sqrt(mesh.entity_measure('cell'))
error = mesh.error(pde.gradient, uh.grad_value, celltype=True)
tmr.send(f'误差计算时间')

kargs = bm.context(node)
bc = bm.array([[1/3, 1/3, 1/3]], **kargs)
m0 = -bm.log(error/h)
for i in range(10):
    m1 = s0.space.project(m0)
    m0 = m1(bc)
    print(m0.shape)
tmr.send(f'投影时间')


@barycentric
def force(bcs, index):
    return -m1.grad_value(bcs, index)

material = LinearElasticMaterial('movingmesh', elastic_modulus=1,
                                 poisson_ratio=0.3)
GD = mesh.geo_dimension()
s1 = LinearElasticityLFEMSolver(s0.space, GD, material, force)
s1.set_corner_disp_zero()
s1.set_normal_disp_zero(domain)
du = s1.solve()


fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
plot = axes.plot_trisurf(node[:, 0], node[:, 1], m1, triangles=cell, cmap='rainbow')
fig.colorbar(plot, ax=axes)


fig = plt.figure()
axes = fig.add_subplot(111)
axes.quiver(node[:, 0], node[:, 1], du[:, 0], du[:, 1])

fig = plt.figure()
axes = fig.add_subplot(121)
mesh.add_plot(axes)
axes = fig.add_subplot(122)
node += du
mesh.add_plot(axes)
plt.show()
