from fealpy.mesh import IntervalMesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh as quad1
from fealpy.old.mesh import QuadrangleMesh as quad2
from fealpy.old.mesh import HexahedronMesh as hexa
from fealpy.old.mesh import TetrahedronMesh as tetra
from fealpy.mesh import TriangleMesh,HexahedronMesh,TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
mesh1 = quad1.from_box(nx=1,ny=1)
mesh2 = quad2.from_box(nx=1,ny=1)
mesh3 = TriangleMesh.from_box(nx=1,ny=1)
mesh4 = HexahedronMesh.from_box(nx=1,ny=1)
mesh5 = hexa.from_box(nx=1,ny=1)
mesh6 = TetrahedronMesh.from_box(nx=1,ny=1)
mesh7 = TetrahedronMesh.from_box(nx=1,ny=1)
p0=1
p1=4
space = LagrangeFESpace(mesh1,p=p1)
cd = [1,2]
Ps = space.prolongation_matrix(cd)
print(len(Ps))
# ipoints = mesh4.interpolation_points(p=p1)
# cell2ipoints = mesh4.cell_to_ipoint(p=p1)

# # 创建一个包含两个子图的图形
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')  # 使用三维绘图轴
# ax2 = fig.add_subplot(122, projection='3d')  # 使用三维绘图轴

# # 绘制第一个子图
# mesh4.add_plot(ax1, cellcolor='g')
# mesh4.find_node(ax1, showindex=True, color='r', marker='o', fontsize=15, fontcolor='r')
# mesh4.find_edge(ax1, showindex=True, color='b', marker='v', fontsize=20, fontcolor='b')
# mesh4.find_cell(ax1, showindex=True, color='k', marker='s', fontsize=25, fontcolor='k')

# # 绘制第二个子图
# mesh4.add_plot(ax2, cellcolor='g')
# mesh4.find_node(ax2, node=ipoints, showindex=True, color='r', marker='o', fontsize=15, fontcolor='r')

# P2 = mesh2.prolongation_matrix(p0,p1)
# print(P2)

# P1 = mesh1.prolongation_matrix(p0,p1)
# print(P1)
# mesh2.prolongation_matrix(p0,p1)

P1=mesh6.prolongation_matrix(p0,p1)
P2 = mesh7.prolongation_matrix(p0,p1)
print(P2.toarray() == P1.toarray())